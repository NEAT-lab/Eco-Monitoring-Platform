import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion
import board
import adafruit_dht
import time

MQTT_SERVER = "140.116.245.211"
MQTT_PORT = 1883

TOPIC = "eco/sensor/room1"

DHT_PIN = board.D4 # GPIO 4

INTERVAL = 10 # seconds

# init dht11
dht11 = adafruit_dht.DHT11(DHT_PIN)

if __name__ == "__main__" : 
    # connect to mqtt server
    client = mqtt.Client(callback_api_version = CallbackAPIVersion.VERSION2)
    client.connect(MQTT_SERVER, MQTT_PORT, 60)
    # loop
    try : 
        while True : 
            # read temperature and humidity
            temperature = None
            humidity = None
            while temperature is None or humidity is None : 
                try : 
                    temperature = dht11.temperature
                    humidity = dht11.humidity
                    if temperature is None or humidity is None : 
                        time.sleep(INTERVAL)
                except : 
                    print("[E] Fail to Read From DHT11 Sensor")
                    time.sleep(INTERVAL)
            # publish humidity to mqtt server (2 digits after decimal point)
            payload = f'{{"temperature": {temperature}, "humidity": {humidity}}}'
            client.publish(TOPIC, payload)
            # print result
            print("[P] Update Humidity to " + payload)
            # delay seconds
            time.sleep(INTERVAL)
    except KeyboardInterrupt : 
        print()
        print("[E] Recieved Keyboard Interrupt")
    finally : 
        print("[P] Process Shut Down")