import cv2
import os
import time
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
rtsp_url = "rtsp://root:pass@221.120.74.49:9664/axis-media/media.amp"
rtsp_url_2 = "rtsp://root:pass@221.120.74.49:9666/axis-media/media.amp"
# rtsp_url = "rtsp://rtspstream:sgxCxxzgOsO2ShXv_F5Jj@zephyr.rtsp.stream/movie"

model = YOLO("pt/yolo11x.pt")

def generate_frames():
    while True:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(" RTSP open failed, retrying in 3s...")
            time.sleep(3)
            continue

        prev_frame_time = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                print(" Frame read failed, reconnecting...")
                cap.release()
                break

            results = model(frame, verbose=False)
            
            # 只保留 bird 類別的框
            bird_frame = frame.copy()
            bird_count = 0
            
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                
                if class_name.lower() == 'bird':  # 只框 bird
                    bird_count += 1
                    # 繪製邊框
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(bird_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    # 標籤
                    cv2.putText(bird_frame, 'bird', (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            
            # 計算 FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time) if (current_time - prev_frame_time) > 0 else 0
            prev_frame_time = current_time
            
            # 顯示 bird 總數
            cv2.putText(bird_frame, f"Birds: {bird_count}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, (0, 255, 0), 2)
            
            # 顯示 FPS
            cv2.putText(bird_frame, f"FPS: {fps:.1f}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, (255, 0, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', bird_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_frames_2():
    while True:
        cap = cv2.VideoCapture(rtsp_url_2, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(" RTSP open failed, retrying in 3s...")
            time.sleep(3)
            continue

        prev_frame_time = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                print(" Frame read failed, reconnecting...")
                cap.release()
                break

            results = model(frame, verbose=False)
            
            # 只保留 bird 類別的框
            bird_frame = frame.copy()
            bird_count = 0
            
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                
                if class_name.lower() == 'bird':  # 只框 bird
                    bird_count += 1
                    # 繪製邊框
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(bird_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 標籤
                    cv2.putText(bird_frame, 'bird', (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 計算 FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time) if (current_time - prev_frame_time) > 0 else 0
            prev_frame_time = current_time
            
            # 顯示 bird 總數
            cv2.putText(bird_frame, f"Birds: {bird_count}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, (0, 255, 0), 2)
            
            # 顯示 FPS
            cv2.putText(bird_frame, f"FPS: {fps:.1f}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, (255, 0, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', bird_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')