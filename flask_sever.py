from flask import Flask, request, render_template, jsonify, send_from_directory, Response, send_file
import os
from ultralytics import YOLO, RTDETR
import cv2
import subprocess
import numpy as np
import base64
import subprocess
from common_parameters import latest_data
import time
from requests.auth import HTTPDigestAuth, HTTPBasicAuth
import requests
import urllib3
import sqlite3
from datetime import datetime
import threading
from collections import deque
import csv

DB_NAME = 'cctv.db'

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

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

generate_frames_model_light = RTDETR("/home/neat/Ron_train/comparison/gradcam/MANUAL_SAVE_sec_01752.pt")
generate_frames_model_night = RTDETR("pt/rtdetr_night.pt")

db_lock = threading.Lock() 
csv_lock = threading.Lock()
current_stat = {
    'hour': datetime.now().strftime("%Y-%m-%d %H:00"),
    'max_total': 0,
    'counts': {0: 0, 1: 0, 2: 0, 3: 0}
}

def get_current_model():
    """在夜間使用夜間模型，白天使用標準模型。"""
    hour = datetime.now().hour
    # 20:00~05:59 走夜間模型
    if hour >= 20 or hour < 6:
        return generate_frames_model_night
    return generate_frames_model_light

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
                        for box in results[0].boxes:
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
                    update_hourly_max(total_count, detect_frame)
                    update_csv_hourly(total_count, counts)

                    current_time = time.time()
                    fps = 1 / (current_time - prev_frame_time) if current_time > prev_frame_time else 0
                    fps_list.append(fps)
                    smooth_fps = sum(fps_list) / len(fps_list)
                    prev_frame_time = current_time
                    
                    # 畫面上方資訊顯示 (自動排列)
                    h, w = detect_frame.shape[:2]
                    f_scale, thick = w / 1400, max(1, int(w / 500))
                    info_data = [
                        (f"cam : {self.camera_name}", (0, 255, 255)),
                        (f"fps : {smooth_fps:.1f}", (255, 0, 0)),
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
            
        time.sleep(0.03) # 限制傳輸 FPS
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


IMAGE_FOLDER = os.path.join("static", "captures")
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

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

def update_csv_hourly(total_count, counts_dict):
    global current_stat
    now = datetime.now()
    hour_str = now.strftime("%Y-%m-%d %H:00")
    
    with csv_lock:
        # 檢查是否跨小時了
        if hour_str != current_stat['hour']:
            # 1. 跨小時了，先把「上一個小時」的結算結果存進 CSV
            save_to_csv(current_stat['hour'], current_stat['max_total'], current_stat['counts'])
            
            # 2. 重置統計資料為新的一小時
            current_stat['hour'] = hour_str
            current_stat['max_total'] = 0
            current_stat['counts'] = {0: 0, 1: 0, 2: 0, 3: 0}

        # 只要目前畫面的數量更多，就覆蓋暫存紀錄 (不論是哪台 cam)
        if total_count > current_stat['max_total']:
            current_stat['max_total'] = total_count
            current_stat['counts'] = counts_dict

def save_to_csv(time_label, total, counts):
    file_path = 'cctv_database.csv'
    header = ['Time', 'Total', 'Anatidae', 'Ardea_cinerea', 'Turtle', 'Nycticorax']
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([time_label, total, counts[0], counts[1], counts[2], counts[3]])
    print(f"[CSV] 已結算並存檔: {time_label} - Max: {total}")

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results/runs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

MODEL_FOLDER = "pt"
available_models = sorted([f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pt")])
models_list = [YOLO(os.path.join(MODEL_FOLDER, f)) for f in available_models]
models = {os.path.join(MODEL_FOLDER, f): m for f, m in zip(available_models, models_list)}

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
rtsp_url_1 = "rtsp://root:pass@221.120.74.49:9664/axis-media/media.amp"
rtsp_url_2 = "rtsp://root:pass@221.120.74.49:9666/axis-media/media.amp"
# --- 設定多台攝影機 ---
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

# html pages
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

# API
@app.route("/get_models")
def get_models():
    return jsonify({"models": available_models})

@app.route("/api/detect", methods=["POST"])
def api_detect():
    total_count = 0
    count_by_class = {}
    
    model_path = request.form.get("model_path")
    print("使用模型:", model_path)
    if model_path in models:
        model = models[model_path]
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

@app.route("/api/panorama", methods=["POST"])
def api_panorama():
    total_count = 0
    count_by_class = {}

    model_path = request.form.get("model_path")
    if model_path in models:
        model = models[model_path]
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
  
@app.route('/api/get_zoom', methods=['GET'])
def get_zoom():
    # 設定攝影機 URL (讀取狀態需使用 query=position)
    url = f'https://221.120.74.49:9661/axis-cgi/com/ptz.cgi'
    params = {'query': 'position', 'camera': 1}

    try:
        # 發送 GET 請求到攝影機
        response = requests.get(
            url,
            params=params,
            auth=HTTPDigestAuth("root", "pass"),
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

    except:
        return jsonify({'error': '無法連接到攝影機'}), 500

@app.route('/api/get_zoom_2', methods=['GET'])
def get_zoom_2():
    # 設定攝影機 URL (讀取狀態需使用 query=position)
    url = f'https://221.120.74.49:9663/axis-cgi/com/ptz.cgi'
    params = {'query': 'position', 'camera': 1}

    try:
        # 發送 GET 請求到攝影機
        response = requests.get(
            url,
            params=params,
            auth=HTTPBasicAuth("root", "pass"),
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

    except:
        return jsonify({'error': '無法連接到攝影機'}), 500
     
@app.route('/api/zoom', methods=['POST'])
def zoom():
    # 獲取前端發送的 JSON 數據
    data = request.get_json()
    zoom_value = data.get('zoom')

    # 發送命令到攝影機
    url = f'https://221.120.74.49:9661/axis-cgi/com/ptz.cgi'
    params = {'zoom': zoom_value, 'camera': 1}

    try:
        response = requests.post(
            url,
            data=params,
            auth=HTTPDigestAuth("root", "pass"),
            verify=False,
            timeout=5
        )
        return jsonify({'status': 'success', 'zoom': zoom_value})
    except:
        return jsonify({'error': '無法連接到攝影機'}), 500
    
@app.route('/api/zoom_2', methods=['POST'])
def zoom_2():
    # 獲取前端發送的 JSON 數據
    data = request.get_json()
    zoom_value = data.get('zoom')

    # 發送命令到攝影機
    url = f'https://221.120.74.49:9663/axis-cgi/com/ptz.cgi'
    params = {'zoom': zoom_value, 'camera': 1}

    try:
        response = requests.post(
            url,
            data=params,
            auth=HTTPBasicAuth("root", "pass"),
            verify=False,
            timeout=5
        )
        return jsonify({'status': 'success', 'zoom': zoom_value})
    except:
        return jsonify({'error': '無法連接到攝影機'}), 500

@app.route('/api/direction_2', methods=['POST'])
def direction_2():
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

    # 發送命令到攝影機
    url = f'https://221.120.74.49:9663/axis-cgi/com/ptz.cgi'

    try:
        response = requests.post(
            url,
            data=params,
            auth=HTTPBasicAuth("root", "pass"),
            verify=False,
            timeout=5
        )
        return jsonify({'status': 'success', 'direction': direction})
    except:
        return jsonify({'error': '無法連接到攝影機'}), 500
    
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