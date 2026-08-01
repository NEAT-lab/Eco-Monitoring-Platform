# mqtt_sever.py 和 flask_sever.py 共用的感測器資料
# 樹莓派 (device_sensor/rpi_publish.py) 讀取 DHT11 後透過 MQTT 發布，
# mqtt_sever.py 訂閱後寫入這裡，flask_sever.py 的 /api/data 再讀出來給前端顯示

latest_data = {
    "temperature": None,
    "humidity": None,
    "timestamp": None
}