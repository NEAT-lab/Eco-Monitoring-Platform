import paho.mqtt.client as mqtt
import time

def on_connect(client, userdata, flags, reason_code, properties=None):
    print("Connected with reason code:", reason_code)
    client.subscribe("eco/sensor/room1")

def on_message(client, userdata, msg):
    print(msg.topic, msg.payload.decode())

esp32 = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

esp32.on_connect = on_connect
esp32.on_message = on_message
esp32.connect("localhost", 1883, 60)
esp32.loop_start()

time.sleep(5)  # 等待連線建立

while True:
    temperature = 27.5  # 直接寫死，也可以改成 random
    humidity = 63.0

    payload = f'{{"temperature": {temperature}, "humidity": {humidity}}}'
    
    esp32.publish("eco/sensor/room1", payload)
    print("Published:", payload)

    time.sleep(5)  # 每 5 秒發一次