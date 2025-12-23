from flask import Flask, request, render_template, jsonify, send_from_directory, Response
import os
from ultralytics import YOLO
import cv2
import subprocess
import numpy as np
import base64
import subprocess
from common_parameters import latest_data
import time
import rtsp_sever

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


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results/runs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

MODEL_FOLDER = "pt"
available_models = sorted([f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pt")])
models_list = [YOLO(os.path.join(MODEL_FOLDER, f)) for f in available_models]
models = {os.path.join(MODEL_FOLDER, f): m for f, m in zip(available_models, models_list)}



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

@app.route('/video_feed')
def video_feed():
    return Response(rtsp_sever.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')