# Web 框架
from flask import Flask, request, render_template, jsonify, send_from_directory, Response, send_file
import os
from dotenv import load_dotenv

# 影像辨識模型與影像處理
from ultralytics import YOLO, RTDETR
import cv2
import numpy as np
from torchvision.ops import nms  

# 攝影機/外部 API 通訊
from requests.auth import HTTPDigestAuth, HTTPBasicAuth
import requests
import urllib3
import subprocess  

# 資料庫與檔案輸出
import sqlite3
import csv
import base64
from sensor_data import latest_data

# 音訊辨識
import io
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import gc

# 通用工具
import time
import random
from datetime import datetime
import threading
from collections import deque

# ========== 環境設定與常數 ==========
load_dotenv()

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

DB_NAME = 'cctv.db'

# 攝影機連線資訊（從 .env 讀取，換攝影機/改密碼只需改 .env）
CAMERA_USER = os.getenv("CAMERA_USER")
CAMERA_PASS = os.getenv("CAMERA_PASS")
CAMERA_HOST = os.getenv("CAMERA_HOST")

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results/runs"
DETECTION_MODEL_FOLDER = "weights/detection"
CCTV_MODEL_FOLDER = "weights/cctv"
IMAGE_FOLDER = os.path.join("static", "captures")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(DETECTION_MODEL_FOLDER, exist_ok=True)
os.makedirs(CCTV_MODEL_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# ========== 共用狀態與鎖（cctv）==========
db_lock = threading.Lock()
csv_lock = threading.Lock()
current_stat = {
    'hour': datetime.now().strftime("%Y-%m-%d %H:00:00"),
    'max_total': 0,
    'counts': {0: 0, 1: 0, 2: 0, 3: 0},
    'weather': None
}

# 簡單的燈光狀態快取 (隨機間隔更新)
light_cache = {'state': False, 'last_update': 0, 'interval': random.randint(180, 360), 'checking': True, 'last_active_period': None}

# ========== 偵測模型載入（detect／panorama）==========
available_models = sorted([f for f in os.listdir(DETECTION_MODEL_FOLDER) if f.endswith(".pt")])
detection_model_objects = [YOLO(os.path.join(DETECTION_MODEL_FOLDER, f)) for f in available_models]
detection_models = {os.path.join(DETECTION_MODEL_FOLDER, f): m for f, m in zip(available_models, detection_model_objects)}

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("已開啟動態顯存分配")
    except RuntimeError as e:
        print(e)

# ========== 音訊模型載入（audio）==========
audio_model = tf.keras.models.load_model('audio_detect/model/best_bird_model.keras')

# ========== CCTV 模型載入（cctv）==========
cctv_model_day = RTDETR(os.path.join(CCTV_MODEL_FOLDER, "rtdetr_day.pt"))
cctv_model_night = RTDETR(os.path.join(CCTV_MODEL_FOLDER, "rtdetr_night.pt"))

# ========== 資料庫（cctv）==========
def init_db():
    # 連接資料庫 (如果檔案不存在，會自動建立)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 執行 SQL 指令建立表格
    # IF NOT EXISTS: 避免重複建立報錯
    # PRIMARY KEY (date, hour): 設定複合主鍵
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hourly_max (
            date TEXT,
            hour INTEGER,
            max_count INTEGER,
            PRIMARY KEY (date, hour)
        )
    ''')

    conn.commit() # 確認執行
    conn.close()  # 關閉連線
    print(f"成功建立資料庫: {DB_NAME}")

def update_hourly_max(current_total_count, frame):
    # 1. 取得當前時間資訊
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d') # 格式: 2023-10-27
    current_hour = now.hour             # 格式: 14 (代表下午兩點)
    filename = f"bird_{date_str}_{current_hour}.jpg"

    with db_lock:
        try:
            with sqlite3.connect(DB_NAME) as conn:
                cursor = conn.cursor()

                # 2. 查詢該小時目前的紀錄
                cursor.execute(
                    'SELECT max_count FROM hourly_max WHERE date = ? AND hour = ?',
                    (date_str, current_hour)
                )
                row = cursor.fetchone()

                save_image = False # 標記是否需要存照片

                if row is None:
                    # 3. 情況 A: 該小時還沒有任何紀錄 -> 直接新增
                    cursor.execute(
                        'INSERT INTO hourly_max (date, hour, max_count) VALUES (?, ?, ?)',
                        (date_str, current_hour, current_total_count)
                    )
                    save_image = True
                    print(f"[{date_str} {current_hour}:00] 新增紀錄: {current_total_count} 隻")

                else:
                    # 4. 情況 B: 該小時已有紀錄 -> 檢查是否打破紀錄
                    existing_max = row[0]
                    if current_total_count > existing_max:
                        cursor.execute(
                            'UPDATE hourly_max SET max_count = ? WHERE date = ? AND hour = ?',
                            (current_total_count, date_str, current_hour)
                        )
                        save_image = True
                        print(f"[{date_str} {current_hour}:00] 更新最大值: {existing_max} -> {current_total_count} 隻")

                conn.commit()

                # 如果有更新紀錄，就直接把照片存到硬碟 (覆蓋舊的)
                if save_image and frame is not None:
                    image_path = os.path.join(IMAGE_FOLDER, filename)
                    cv2.imwrite(image_path, frame)
                    print(f"已更新最大值照片: {filename}")

        except Exception as e:
            print(f"資料庫更新失敗: {e}")

# ========== 天氣與 CSV 記錄工具（cctv）==========
def get_selected_weather(lat, lon):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        "wind_direction_10m,precipitation,cloud_cover,weathercode"
        "&daily=sunrise,sunset"
        "&timezone=auto"
    )

    weather_map = {
        0: "clear",          # 晴
        1: "mostly clear",   # 晴到多雲
        2: "partly cloudy",  # 多雲
        3: "overcast",       # 陰天
        45: "fog",           # 霧
        48: "fog",

        # 小雨系列（毛毛雨、連續小雨、短暫小陣雨）
        51: "light rain", 53: "light rain", 55: "light rain",
        61: "light rain", 80: "light rain",

        # 大雨系列（中雨、連續大雨、強烈陣雨）
        63: "heavy rain", 65: "heavy rain",
        81: "heavy rain", 82: "heavy rain",

        # 雷陣雨（通常伴隨大暴雨與雷聲）
        95: "thunderstorm", 96: "thunderstorm", 97: "thunderstorm",

        # 降雪（台灣平地遇不到，防錯保留）
        71: "snow", 73: "snow", 75: "snow", 77: "snow", 56: "snow", 57: "snow", 66: "snow", 67: "snow", 85: "snow", 86: "snow"
    }

    for i in range(3):  # 最多重試3次
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()

                current = data["current"]
                daily = data["daily"]

                print(f"successfully weather data: {data}")

                # 基本欄位
                temp = current["temperature_2m"]
                rh = current["relative_humidity_2m"]
                dew_point = round(temp - ((100 - rh) / 5), 2)

                code = current.get("weathercode", -1)

                try:
                    sunrise_raw = daily["sunrise"][0]
                    sunset_raw = daily["sunset"][0]
                    sunrise_time = sunrise_raw.split('T')[1] if 'T' in sunrise_raw else sunrise_raw[11:16]
                    sunset_time = sunset_raw.split('T')[1] if 'T' in sunset_raw else sunset_raw[11:16]
                except Exception:
                    sunrise_time = ""
                    sunset_time = ""

                return {
                    "Description": weather_map.get(code, str(code)),
                    "Temperature (°C)": temp,
                    "Dew Point (°C)": dew_point,
                    "Humidity (%)": rh,
                    # 修正：Open-Meteo 預設為 km/h，在此轉換為 m/s
                    "Wind Speed (m/s)": round(current["wind_speed_10m"] / 3.6, 2),
                    "Wind Direction (°)": current.get("wind_direction_10m"),
                    "Cloud Coverage (%)": current.get("cloud_cover"),
                    "Rainfall Last 1h (mm)": current.get("precipitation", 0),
                    "Sunrise": sunrise_time,
                    "Sunset": sunset_time
                }

            else:
                print(f"天氣 API 請求失敗 (狀態碼: {response.status_code})，重試中... ({i+1}/3)")

        except Exception as e:
            print(f"天氣 API 請求錯誤: {e}，重試中... ({i+1}/3)")

        if i < 2:
            time.sleep(2)

    return None

def update_csv_hourly(total_count, counts_dict):
    global current_stat
    now = datetime.now()
    hour_str = now.strftime("%Y-%m-%d %H:00:00")

    with csv_lock:
        # 檢查是否跨小時了
        if hour_str != current_stat['hour']:
            # 1. 跨小時了，先把「上一個小時」的結算結果存進 CSV
            save_to_csv(current_stat['hour'], current_stat['max_total'], current_stat['counts'], current_stat['weather'])

            # 2. 重置統計資料為新的一小時
            current_stat['hour'] = hour_str
            current_stat['max_total'] = 0
            current_stat['counts'] = {0: 0, 1: 0, 2: 0, 3: 0}
            current_stat['weather'] = None

        # 只要目前畫面的數量更多，就覆蓋暫存紀錄 (不論是哪台 cam)
        if total_count > current_stat['max_total']:
            current_stat['max_total'] = total_count
            current_stat['counts'] = counts_dict

def save_to_csv(time_label, total, counts, weather):
    file_path = 'cctv_database.csv'
    header = ['Time', 'Total', 'Anatidae', 'Ardea_cinerea', 'Turtle', 'Nycticorax',
              'Description', 'Temperature (°C)', 'Dew Point (°C)', 'Humidity (%)',
              'Wind Speed (m/s)', 'Wind Direction (°)', 'Cloud Coverage (%)', 'Rainfall Last 1h (mm)', 'Sunrise', 'Sunset']

    file_exists = os.path.isfile(file_path)

    if weather is None:
        print("Weather data is None, using default values.")
        weather = {
            "Description": "",
            "Temperature (°C)": "",
            "Dew Point (°C)": "",
            "Humidity (%)": "",
            "Wind Speed (m/s)": "",
            "Wind Direction (°)": "",
            "Cloud Coverage (%)": "",
            "Rainfall Last 1h (mm)": "",
            "Sunrise": "",
            "Sunset": ""
        }

    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([time_label, total, counts[0], counts[1], counts[2], counts[3],
                         weather["Description"], weather["Temperature (°C)"], weather["Dew Point (°C)"],
                            weather["Humidity (%)"], weather["Wind Speed (m/s)"], weather["Wind Direction (°)"],
                            weather["Cloud Coverage (%)"], weather["Rainfall Last 1h (mm)"], weather["Sunrise"], weather["Sunset"]])

    print(f"[CSV] 結算成功: {time_label} (Max:{total} 當時天氣:{weather['Description']})")

# ========== 影片轉碼工具（detect）==========
def to_mp4(input_path, output_path=None):
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".mp4"

    cmd = [
        "ffmpeg",
        "-y",               # 覆蓋已存在檔案
        "-i", input_path,   # 輸入影片
        "-c:v", "libx264",  # 視訊編碼
        "-preset", "fast",
        "-pix_fmt", "yuv420p",  # Chrome 可播放
        "-c:a", "aac",          # 音訊編碼
        "-b:a", "128k",          # 音訊 bitrate
        output_path
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

# ========== 攝影機串流核心（cctv）==========
def get_current_model():
    """根據燈光狀態選擇模型：燈亮=夜間模型，燈暗=白天模型"""
    global light_cache
    current_time = time.time()
    current_hour = datetime.now().hour

    current_period = None
    expected_state = None

    if 5 <= current_hour < 7:
        current_period = "morning"
        expected_state = False
    elif 17 <= current_hour < 20:
        current_period = "evening"
        expected_state = True

    if current_period != light_cache['last_active_period']:
        light_cache['checking'] = True
        light_cache['last_active_period'] = current_period
        print(f"[Model] 時段變更，重置檢查狀態。當前時段: {current_period}")

    if current_time - light_cache['last_update'] > light_cache['interval'] and (5 <= current_hour < 7 or 17 <= current_hour < 20) and current_period and light_cache['checking']:
        # 檢查燈光狀態
        url = f'https://{CAMERA_HOST}:9663/axis-cgi/lightcontrol.cgi'
        payload = {
            "apiVersion": "1.4",
            "context": "state_check",
            "method": "getLightInformation"
        }

        try:
            response = requests.post(
                url,
                json=payload,
                auth=HTTPBasicAuth(CAMERA_USER, CAMERA_PASS),
                verify=False,
                timeout=3
            )

            res_data = response.json()
            is_light_on = res_data.get('data', {}).get('items', [{}])[0].get('lightState', False)
            light_cache['state'] = is_light_on
            light_cache['last_update'] = current_time
            light_cache['interval'] = random.randint(180, 360)
            print(f"[Model] 燈光狀態更新: {'夜間模式' if is_light_on else '白天模式'}")

            if is_light_on == expected_state:
                light_cache['checking'] = False
                print(f"🎉 [Model] 燈光已達到預期狀態 ({'亮' if expected_state else '暗'})，鎖定 {current_period} 時段，停止後續 API 請求。")

        except Exception as e:
            print(f"[Model] 燈光狀態檢查失敗: {e}")
            # 失敗時使用預設值 (白天模式)
            pass

    # 根據燈光狀態選擇模型
    if light_cache['state']:
        return cctv_model_night  # 燈亮 = 夜間
    else:
        return cctv_model_day  # 燈暗 = 白天

class CameraStream:
    def __init__(self, rtsp_url, camera_name):
        self.rtsp_url = rtsp_url
        self.camera_name = camera_name # 用於辨識是哪一台攝影機
        self.frame_bytes = None
        self.lock = threading.Lock()
        self.stopped = False

        # 資料集收集相關
        self.last_dataset_save_time = time.time()  # 上次保存的時間
        self.dataset_save_interval = 1440  # 每1440秒保存一次（每小時3張）
        self.dataset_folder = "dataset"  # 資料集主文件夾
        self.total_count_history = deque(maxlen=90)
        self.hour_str = datetime.now().strftime("%Y-%m-%d %H:00:00") # 當前小時的標籤
        self.interval = random.randint(300, 600)
        self.frame_interval = 0.03  # video_feed 傳輸節流間隔，決定瀏覽器實際能收到新畫面的速度上限

        # 啟動背景執行緒
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        """
        生產者：每一台攝影機都有自己獨立的 update 迴圈
        """
        print(f" [Stream-{self.camera_name}] Background thread started.")
        while not self.stopped:
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                print(f" [Stream-{self.camera_name}] RTSP failed. Retrying in 3s...")
                time.sleep(3)
                continue

            print(f" [Stream-{self.camera_name}] Connected.")
            prev_frame_time = time.time()
            fps_list = deque(maxlen=240)
            last_call_weather_time = time.time()

            while not self.stopped:
                success, frame = cap.read()
                if not success:
                    print(f" [Stream-{self.camera_name}] Read failed, reconnecting...")
                    break

                try:
                    model = get_current_model()
                    results = model.predict(frame, conf=0.4, iou=0.5, verbose=False)

                    detect_frame = frame.copy()
                    total_count = 0
                    counts = {0: 0, 1: 0, 2: 0, 3: 0} # 對應 Anatidae, Ardea_cinerea, Turtle, Nycticorax

                    colors = {
                        0: (255, 0, 0),      # Anatidae
                        1: (231, 224, 87),   # Ardea_cinerea
                        2: (29, 147, 123),   # Turtle
                        3: (255, 0, 255)     # Nycticorax
                    }

                    if results and results[0].boxes:
                        # --- 核心手動 NMS 邏輯 (僅加這幾行) ---
                        # 取得所有框的索引，iou_threshold 可依需求調整 (0.3~0.5)
                        keep_idx = nms(results[0].boxes.xyxy, results[0].boxes.conf, iou_threshold=0.3)
                        filtered_boxes = [results[0].boxes[i] for i in keep_idx]

                        for box in filtered_boxes:
                            total_count += 1
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf, cls = float(box.conf[0]), int(box.cls[0])

                            # 更新分類統計
                            if cls in counts:
                                counts[cls] += 1

                            # 繪製邊框
                            color = colors.get(cls, (0, 255, 255))
                            cv2.rectangle(detect_frame, (x1, y1), (x2, y2), color, 2)

                            # 繪製信心度標籤 (含背景)
                            text = f"{conf:.2f}"
                            (w, h), b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(detect_frame, (x1, y1 - h - b - 5), (x1 + w, y1), color, -1)
                            cv2.putText(detect_frame, text, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # 更新統計與 FPS
                    self.total_count_history.append(total_count)
                    if len(self.total_count_history) == self.total_count_history.maxlen:
                        occurrence_count = list(self.total_count_history).count(total_count)
                        stability_rate = occurrence_count / len(self.total_count_history)

                        if stability_rate >= 0.8 or self.hour_str != datetime.now().strftime("%Y-%m-%d %H:00:00"):
                            update_hourly_max(total_count, detect_frame)
                            update_csv_hourly(total_count, counts)
                            if self.hour_str != datetime.now().strftime("%Y-%m-%d %H:00:00"):
                                self.hour_str = datetime.now().strftime("%Y-%m-%d %H:00:00")
                                last_call_weather_time = time.time()
                        else:
                            pass

                    current_time = time.time()
                    if current_time - last_call_weather_time >= self.interval:
                        with csv_lock:
                            # 鎖內重新確認，避免兩台攝影機同時通過檢查而重複呼叫天氣 API
                            if current_stat["weather"] is None:
                                current_stat["weather"] = get_selected_weather(23.050288847837354, 120.14645308748925)
                        last_call_weather_time = current_time
                        self.interval = random.randint(300, 600)

                    fps = 1 / (current_time - prev_frame_time) if current_time > prev_frame_time else 0
                    fps_list.append(fps)
                    smooth_fps = sum(fps_list) / len(fps_list)
                    prev_frame_time = current_time

                    # 顯示用 fps：不能超過 video_feed 的傳輸節流上限，避免跟實際畫面流暢度脫節
                    display_fps = min(smooth_fps, 1 / self.frame_interval)

                    # 畫面上方資訊顯示 (自動排列)
                    h, w = detect_frame.shape[:2]
                    f_scale, thick = w / 1400, max(1, int(w / 500))
                    info_data = [
                        (f"cam : {self.camera_name}", (0, 255, 255)),
                        (f"fps : {display_fps:.1f}", (255, 0, 0)),
                        (f"total : {total_count}", (0, 255, 0)),
                        (f"Anatidae : {counts[0]}", (255, 0, 0)),
                        (f"Ardea_cinerea : {counts[1]}", (231, 224, 87)),
                        (f"Turtle : {counts[2]}", (0, 255, 0)),
                        (f"Nycticorax : {counts[3]}", (255, 0, 255))
                    ]

                    for i, (txt, clr) in enumerate(info_data):
                        cv2.putText(detect_frame, txt, (10, int((0.08 + i*0.05) * h)),
                                    cv2.FONT_HERSHEY_SIMPLEX, f_scale, clr, thick)

                    # 編碼與保存
                    ret, buffer = cv2.imencode('.jpg', detect_frame)
                    if ret:
                        with self.lock:
                            self.frame_bytes = buffer.tobytes()

                        if current_time - self.last_dataset_save_time >= self.dataset_save_interval:
                            self.save_dataset_frame(frame)
                            self.last_dataset_save_time = current_time

                except Exception as e:
                    print(f" [Stream-{self.camera_name}] Error: {e}")
                    pass

            cap.release()

    def get_frame(self):
        with self.lock:
            return self.frame_bytes

    def save_dataset_frame(self, frame):
        """
        保存幀到 dataset 文件夾
        文件名格式: camera_YYYY_MM_DD_HH_MM.jpg
        """
        try:
            now = datetime.now()

            # 創建 dataset 文件夾
            os.makedirs(self.dataset_folder, exist_ok=True)

            # 生成文件名：camera_YYYY_MM_DD_HH_MM.jpg
            filename = f"{self.camera_name}_{now.strftime('%Y_%m_%d_%H_%M')}.jpg"
            file_path = os.path.join(self.dataset_folder, filename)

            # 保存圖像
            cv2.imwrite(file_path, frame)
            print(f"[Dataset] 已保存: {file_path}")

        except Exception as e:
            print(f"[Dataset] 保存失敗 [{self.camera_name}]: {e}")


def generate_frames(camera_id):
    """
    生成器現在接收 camera_id 參數
    """
    if camera_id not in streams:
        return None

    active_stream = streams[camera_id]

    while True:
        frame = active_stream.get_frame()

        if frame is None:
            time.sleep(0.1)
            continue

        time.sleep(active_stream.frame_interval) # 限制傳輸 FPS

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ========== Flask App 建立（共用）==========
app = Flask(__name__)

# ========== 攝影機啟動（cctv）==========
rtsp_url_1 = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_HOST}:9664/axis-media/media.amp"
rtsp_url_2 = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_HOST}:9666/axis-media/media.amp"
CAMERAS_CONFIG = {
    "cam1": rtsp_url_1,
    "cam2": rtsp_url_2
}

# 儲存所有串流物件的字典
streams = {}

# 初始化所有攝影機
print("Initializing cameras...")
for cam_name, url in CAMERAS_CONFIG.items():
    streams[cam_name] = CameraStream(url, cam_name)

# PTZ 攝影機設定：每台攝影機各自的控制網址與認證方式
PTZ_CONFIG = {
    "cam1": {"url": f"https://{CAMERA_HOST}:9661/axis-cgi/com/ptz.cgi", "auth": HTTPDigestAuth(CAMERA_USER, CAMERA_PASS)},
    "cam2": {"url": f"https://{CAMERA_HOST}:9663/axis-cgi/com/ptz.cgi", "auth": HTTPBasicAuth(CAMERA_USER, CAMERA_PASS)}
}

# ========== 路由：頁面（共用）==========
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/cctv")
def cctv():
    return render_template("cctv.html")

@app.route("/panorama")
def panorama():
    return render_template("panorama.html")

@app.route("/audio")
def audio():
    return render_template("audio.html")

# ========== 路由：模型清單（detect／panorama）==========
@app.route("/get_models")
def get_models():
    return jsonify({"models": available_models})

# ========== 路由：偵測 API（detect）==========
@app.route("/api/detect", methods=["POST"])
def api_detect():
    total_count = 0
    count_by_class = {}

    model_path = request.form.get("model_path")
    print("使用模型:", model_path)
    if model_path in detection_models:
        model = detection_models[model_path]
    else:
        return jsonify({"type": "error", "message": "Model not found."}), 400

    file = request.files["file"]
    filename = file.filename.lower()
    ext = os.path.splitext(filename)[1]

    if ext in [".jpg",".jpeg",".png",".bmp",".webp"]:

        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        results = model(img)[0]

        for box in results.boxes:
            total_count += 1
            cls = int(box.cls)
            class_name = results.names[cls]
            conf = float(box.conf)
            # 計數
            if class_name not in count_by_class:
                count_by_class[class_name] = 1
            else:
                count_by_class[class_name] += 1

            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            label = f"{results.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"type": "image", "image": img_base64, "total_count": total_count, "count_by_class": count_by_class})

    elif ext in [".mp4",".avi",".mov",".mkv",".webm"]:
        ts = time.strftime("%Y%m%d_%H%M%S")
        file.filename = f"{ts}_{file.filename}"
        input_path = f"{UPLOAD_FOLDER}/{file.filename}"

        file.save(input_path)

        # 先暫存 avi 檔的路徑 (因為 YOLO 會自動存成 avi)
        basename = os.path.splitext(file.filename)[0]
        new_filename = f"{basename}.avi"

        # YOLO 偵測（可直接用model.predict也行）
        model.predict(input_path, save=True, project=RESULT_FOLDER, name="./", exist_ok=True)

        #  取得 YOLO 存出的影片位置
        processed_video = f"{RESULT_FOLDER}/{new_filename}"
        mp4_processed_video = to_mp4(processed_video)
        if os.path.exists(processed_video):
            os.remove(processed_video)

        return jsonify({"type": "video", "video_url": f"/video/{os.path.basename(mp4_processed_video)}"})

    else:
        return jsonify({"type": "error", "message": "Unsupported file format."})

@app.route("/video/<filename>")
def video(filename):
    return send_from_directory(f"{RESULT_FOLDER}", filename)

# ========== 路由：全景 API（panorama）==========
@app.route("/api/panorama", methods=["POST"])
def api_panorama():
    total_count = 0
    count_by_class = {}

    model_path = request.form.get("model_path")
    if model_path in detection_models:
        model = detection_models[model_path]
    else:
        return jsonify({"type": "error", "message": "Model not found."}), 400

    file = request.files["file"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    file.filename = f"{ts}_{file.filename}"
    filename = file.filename.lower()
    ext = os.path.splitext(filename)[1]

    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        return jsonify({"type": "error", "message": "Unsupported video format."}), 400

    input_path = f"{UPLOAD_FOLDER}/{file.filename}"
    output_path = f"{RESULT_FOLDER}/{os.path.splitext(file.filename)[0]}.jpg"
    file.save(input_path)

    # ==================== 步驟1: 全景拼接 ====================
    try:
        print(f"正在讀取影片: {input_path}")
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            return jsonify({"type": "error", "message": "無法開啟影片檔案"}), 500

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = 20
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()

        if len(frames) < 2:
            return jsonify({"type": "error", "message": "提取的幀數不足"}), 500

        # 拼接全景圖
        stitcher = cv2.Stitcher_create()
        status, img = stitcher.stitch(frames)

        if status != cv2.Stitcher_OK:
            # 簡單拼接
            height = frames[0].shape[0]
            resized_frames = []
            for frame in frames:
                if frame.shape[0] != height:
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    new_width = int(height * aspect_ratio)
                    frame = cv2.resize(frame, (new_width, height))
                resized_frames.append(frame)
            img = np.hstack(resized_frames)

    except Exception as e:
        return jsonify({"type": "error", "message": f"全景拼接失敗: {str(e)}"}), 500

    # ==================== 步驟2: 使用與 /predict 相同的辨識邏輯 ====================
    try:
        results = model(img)[0]

        for box in results.boxes:
            total_count += 1
            cls = int(box.cls)
            class_name = results.names[cls]
            conf = float(box.conf)

            if class_name not in count_by_class:
                count_by_class[class_name] = 1
            else:
                count_by_class[class_name] += 1

            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            label = f"{results.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(output_path, img)
        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        # if os.path.exists(input_path):
        #     os.remove(input_path)

        return jsonify({"type": "image", "image": img_base64, "total_count": total_count, "count_by_class": count_by_class})

    except Exception as e:
        return jsonify({"type": "error", "message": f"物件辨識失敗: {str(e)}"}), 500

# ========== 路由：音訊 API（audio）==========
@app.route('/api/audio', methods=['POST'])
def api_audio():
    if 'file' not in request.files:
        return jsonify({"type": "error", "message": "未偵測到上傳檔案"}), 400

    file = request.files['file']

    # 設定參數
    SR = 22050
    DURATION = 2.0
    TARGET_LEN = int(SR * DURATION)
    N_MELS = 224

    try:
        # --- 檔案大小與長度初步檢查 ---
        audio_data = file.read()
        if len(audio_data) > 15 * 1024 * 1024: # 限制 15MB
            return jsonify({"type": "error", "message": "檔案過大，請限制在 15MB 以內"}), 400

        # 直接從記憶體讀取音訊
        audio_stream = io.BytesIO(audio_data)
        y, sr = librosa.load(audio_stream, sr=SR)
        total_sec = librosa.get_duration(y=y, sr=sr)

        if total_sec > 60: # 限制 60 秒
            return jsonify({"type": "error", "message": "音訊過長，請限制在 60 秒以內"}), 400

        results_data = {"time": [], "p1": [], "p2": [], "p3": []}

        # --- 核心分析迴圈 ---
        for start_sample in range(0, len(y), TARGET_LEN):
            y_chunk = y[start_sample : start_sample + TARGET_LEN]
            if len(y_chunk) < TARGET_LEN:
                y_chunk = np.pad(y_chunk, (0, TARGET_LEN - len(y_chunk)))

            # 生成頻譜數據
            S = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=N_MELS, hop_length=196)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # 繪製臨時頻譜圖 (存於記憶體)
            img_buf = io.BytesIO()
            fig_spec, ax_spec = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            ax_spec.axis('off')
            librosa.display.specshow(S_dB, sr=sr, cmap='viridis', ax=ax_spec)
            plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0)

            # 關鍵：立即關閉畫布釋放記憶體
            plt.close(fig_spec)
            fig_spec.clf()

            # AI 預測
            img_buf.seek(0)
            img = tf.keras.utils.load_img(img_buf, target_size=(224, 224))
            img_array = np.expand_dims(tf.keras.utils.img_to_array(img), 0)
            preds = audio_model.predict(img_array, verbose=0)[0]

            results_data["time"].append(round(start_sample / sr, 2))
            results_data["p1"].append(float(preds[0] * 100))
            results_data["p2"].append(float(preds[1] * 100))
            results_data["p3"].append(float(preds[2] * 100))

        # --- 生成最終統計圖表 ---
        final_plot_buf = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.plot(results_data["time"], results_data["p1"], label='Anatidae', color='#1f77b4', linewidth=2)
        plt.plot(results_data["time"], results_data["p2"], label='Ardea cinerea', color='#d62728', linewidth=2)
        plt.plot(results_data["time"], results_data["p3"], label='None (Background)', color='#7f7f7f', linestyle='--')

        plt.title(f"Bird Detection Report: {file.filename}")
        plt.xlabel("Time (Seconds)")
        plt.ylabel("Confidence (%)")
        plt.ylim(-5, 105)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(final_plot_buf, format='png')
        plt.close('all') # 關閉所有剩餘畫布

        final_plot_buf.seek(0)
        img_base64 = base64.b64encode(final_plot_buf.read()).decode("utf-8")

        # --- 釋放資源 ---
        gc.collect()

        return jsonify({
            "type": "image",
            "image": img_base64,
            "filename": file.filename,
            "analysis_data": results_data
        })

    except Exception as e:
        plt.close('all')
        print(f"❌ 錯誤: {e}")
        return jsonify({"type": "error", "message": str(e)}), 500

# ========== 路由：即時串流／資料 API（cctv）==========
@app.route("/api/data")
def api_data():
    return jsonify(latest_data)

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """
    動態路由：根據 URL 的 camera_id 決定回傳哪個串流
    例如: /video_feed/cam1 或 /video_feed/cam2
    """
    if camera_id not in streams:
        return "Camera not found", 404

    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ========== 路由：PTZ 控制 API（cctv）==========
@app.route('/api/get_zoom/<camera_id>', methods=['GET'])
def get_zoom(camera_id):
    if camera_id not in PTZ_CONFIG:
        return jsonify({'error': '找不到攝影機'}), 404
    config = PTZ_CONFIG[camera_id]

    # 讀取狀態需使用 query=position
    params = {'query': 'position', 'camera': 1}

    try:
        # 發送 GET 請求到攝影機
        response = requests.get(
            config['url'],
            params=params,
            auth=config['auth'],
            verify=False,
            timeout=5
        )

        # 解析回傳的文字資料 (尋找 zoom=...)
        zoom_value = None
        for line in response.text.splitlines():
            if line.startswith('zoom='):
                zoom_value = line.split('=')[1]
                break

        # 根據解析結果回傳 JSON
        if zoom_value:
            return jsonify({'status': 'success', 'zoom': zoom_value})
        else:
            return jsonify({'error': '找不到 Zoom 數值'}), 500

    except Exception as e:
        print(f"[PTZ] get_zoom 失敗 [{camera_id}]: {e}")
        return jsonify({'error': '無法連接到攝影機'}), 500

@app.route('/api/zoom/<camera_id>', methods=['POST'])
def zoom(camera_id):
    if camera_id not in PTZ_CONFIG:
        return jsonify({'error': '找不到攝影機'}), 404
    config = PTZ_CONFIG[camera_id]

    # 獲取前端發送的 JSON 數據
    data = request.get_json()
    zoom_value = data.get('zoom')
    params = {'zoom': zoom_value, 'camera': 1}

    try:
        requests.post(
            config['url'],
            data=params,
            auth=config['auth'],
            verify=False,
            timeout=5
        )
        return jsonify({'status': 'success', 'zoom': zoom_value})
    except Exception as e:
        print(f"[PTZ] zoom 失敗 [{camera_id}]: {e}")
        return jsonify({'error': '無法連接到攝影機'}), 500

@app.route('/api/direction/<camera_id>', methods=['POST'])
def direction(camera_id):
    if camera_id not in PTZ_CONFIG:
        return jsonify({'error': '找不到攝影機'}), 404
    config = PTZ_CONFIG[camera_id]

    # 獲取方向
    data = request.get_json()
    direction = data.get('direction')

    # 根據方向決定參數
    params = {'camera': 1}
    if direction == 'up':
        params['tilt'] = 30
    elif direction == 'down':
        params['tilt'] = -30
    elif direction == 'left':
        params['pan'] = -45
    elif direction == 'right':
        params['pan'] = 45

    try:
        requests.post(
            config['url'],
            data=params,
            auth=config['auth'],
            verify=False,
            timeout=5
        )
        return jsonify({'status': 'success', 'direction': direction})
    except Exception as e:
        print(f"[PTZ] direction 失敗 [{camera_id}]: {e}")
        return jsonify({'error': '無法連接到攝影機'}), 500

# ========== 路由：統計與下載 API（cctv）==========
@app.route('/api/daily_stats')
def get_daily_stats():
    """
    API: 根據請求的日期，回傳該日 0~23 點的每小時最大鳥類數量。
    參數: date (格式 YYYY-MM-DD)，若無參數則預設為今天。
    """
    # 取得前端傳來的日期參數，如果沒傳就用今天
    query_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))

    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            # 查詢該日期的所有紀錄
            cursor.execute('SELECT hour, max_count FROM hourly_max WHERE date = ?', (query_date,))
            rows = cursor.fetchall()

            # 初始化一個長度為 24 的陣列，預設值為 0
            # index 0 代表 00:00, index 23 代表 23:00
            hourly_data = [0] * 24

            # 將資料庫查到的數據填入對應的小時
            for hour, count in rows:
                if 0 <= hour < 24:
                    hourly_data[hour] = count

            return jsonify({
                'date': query_date,
                'data': hourly_data
            })

    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_csv')
def download_csv():
    file_path = 'cctv_database.csv'

    if os.path.exists(file_path):
        # as_attachment=True 會強制瀏覽器下載而不是直接打開
        # download_name 可以自訂使用者下載後看到的名字
        return send_file(
            file_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"Biological statistics.csv"
        )
    else:
        return "檔案還沒生成，請稍候再試", 404
