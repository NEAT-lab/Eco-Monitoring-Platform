import json
import time
import paho.mqtt.client as mqtt
from sensor_data import latest_data

# ---------- MQTT ----------
MQTT_BROKER = "localhost"   # 或 EMQX IP
MQTT_PORT = 1883
MQTT_TOPIC = "eco/sensor/room1"

def on_connect(client, userdata, flags, reason_code, properties=None):
    print("MQTT connected:", reason_code)
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    # print("MQTT recv:", payload)

    try:
        data = json.loads(payload)
        latest_data["temperature"] = data.get("temperature")
        latest_data["humidity"] = data.get("humidity")
        latest_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print("JSON parse error:", e)

def mqtt_thread():
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2
    )
    client.on_connect = on_connect
    client.on_message = on_message
    
    time.sleep(5)
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()
    