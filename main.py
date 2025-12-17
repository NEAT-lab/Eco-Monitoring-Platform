import flask_sever
import mqtt_sever
import threading

if __name__ == "__main__":
    threading.Thread(target=mqtt_sever.mqtt_thread, daemon=True).start()

    flask_sever.app.run(debug=False, host="0.0.0.0")
    