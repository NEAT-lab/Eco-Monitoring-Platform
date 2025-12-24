import cv2
import os
import time
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
rtsp_url = "rtsp://rtspstream:sgxCxxzgOsO2ShXv_F5Jj@zephyr.rtsp.stream/movie"

model = YOLO("pt/yolo11n.pt")

def generate_frames():
    while True:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(" RTSP open failed, retrying in 3s...")
            time.sleep(3)
            continue

        while True:
            success, frame = cap.read()
            if not success:
                print(" Frame read failed, reconnecting...")
                cap.release()
                break  # 重新建立 RTSP

            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



