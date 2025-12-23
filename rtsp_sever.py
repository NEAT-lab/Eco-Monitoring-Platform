import cv2
import os
import time
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


# RTSP URL（強制 TCP）
rtsp_url = "rtsp://rtspstream:sgxCxxzgOsO2ShXv_F5Jj@zephyr.rtsp.stream/movie"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

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

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



