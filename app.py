from flask import Flask, request, render_template, jsonify, send_from_directory, send_file
import os
from ultralytics import YOLO
import cv2
import subprocess
import numpy as np
import base64
import subprocess
import os

def to_mp4(input_path, output_path=None):
    """
    將任意影片轉成 MP4，Chrome 可播放。
    input_path: 原始影片路徑（avi/mp4/mkv/mov/...）
    output_path: 輸出路徑（可選），預設跟 input_path 同名但副檔名改為 .mp4
    """
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
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")  # 你要換YOLOv5/v10都可

count = 0

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict",methods=["POST"])
def predict():
    global count
    count = 0
    file = request.files["file"]
    filename = file.filename.lower()
    ext = os.path.splitext(filename)[1]
    
    if ext in [".jpg",".jpeg",".png",".bmp",".webp"]:

        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        results = model(img)[0]

        for box in results.boxes:
            count += 1
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cls = int(box.cls)
            conf = float(box.conf)

            label = f"{results.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"type": "image", "image": img_base64, "count": count})
    
    elif ext in [".mp4",".avi",".mov",".mkv",".webm"]:
        input_path = f"{UPLOAD_FOLDER}/{file.filename}"
        # output_path = f"{RESULT_FOLDER}/result.mp4"

        file.save(input_path)
        
        basename = os.path.splitext(file.filename)[0]
        new_filename = f"{basename}.avi"
        
        # YOLO 偵測（可直接用model.predict也行）
        model.predict(input_path, save=True, project=RESULT_FOLDER, name="runs", exist_ok=True)

        #  取得 YOLO 存出的影片位置
        processed_video = f"results/runs/{new_filename}"
        mp4_processed_video = to_mp4(processed_video)
        if os.path.exists(processed_video):
            os.remove(processed_video)
        
        return jsonify({"type": "video", "video_url": f"/video/{os.path.basename(mp4_processed_video)}"})

    else:
        return jsonify({"type": "error", "message": "Unsupported file format."})
    
@app.route("/video/<filename>")
def video(filename):
    return send_from_directory(f"{RESULT_FOLDER}/runs", filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")