import os
import cv2
import base64
import threading
import time
import requests
import json
import numpy as np
from datetime import datetime
from flask import Flask, render_template
from flask_socketio import SocketIO
from camera import Camera
from gait import GaitProcessor

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Shared state between threads
state = {
    "background_init": False,
    "reset_requested": False,
    "frame_counter": 0,  # Frame counter to track frames processed
    "gei_buffer": None,  # To store the latest GEI image
    "last_saved_gei": None,  # To store the last saved GEI for comparison
}

# Define save directory for GEIs
SAVE_DIR = "saved_geis"

# Save GEI every 150 frames
FRAME_INTERVAL = 30


def is_different(gei_a, gei_b, threshold=5.0):
    if gei_a is None or gei_b is None:
        return True

    if gei_a.shape != gei_b.shape:
        return True

    diff = np.linalg.norm(gei_a.astype("float32") - gei_b.astype("float32"))
    return diff > threshold


def gei_save_thread():
    while True:
        if state["frame_counter"] >= FRAME_INTERVAL:
            current_gei = state["gei_buffer"]
            last_saved_gei = state["last_saved_gei"]

            if current_gei is not None:
                # Check if GEI has changed since last save
                if is_different(current_gei, last_saved_gei):
                    print("GEI has changed. Verifying...")

                    _, buffer = cv2.imencode(".jpg", current_gei)
                    gei_encoded = base64.b64encode(buffer).decode("utf-8")

                    state["last_saved_gei"] = current_gei

                    try:
                        response = requests.post(
                            "http://localhost:5001/api/verify",
                            json={"gei": gei_encoded},
                        )

                        data = json.loads(response.content)
                        print(data)

                        socketio.emit("status", data)

                        print(data["confidence"])

                    except Exception as e:
                        print(f"Failed to save GEI: {e}")

                else:
                    print("GEI unchanged. Skipping...")

            # Reset frame counter
            state["frame_counter"] = 0

        # socketio.emit(
        #     "status",
        #     {
        #         "person": str(random.randint(0, 100000)),
        #         "access": random.choice([True, False]),
        #     },
        # )

        time.sleep(1)


def main():
    cam = Camera()
    cam.start()

    processor = GaitProcessor(buffer_size=30)

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue

            # Handle reset from UI
            if state["reset_requested"]:
                print("Reset requested from frontend. Reinitializing background...")
                state["background_init"] = False
                processor.reset_background()  # You need to define this method in GaitProcessor
                state["reset_requested"] = False

            if not state["background_init"]:
                if processor.initialise_backround(frame):
                    state["background_init"] = True
                    print("Background initialized!")
                continue

            silhouette, gei = processor.process_frame(frame)

            state["frame_counter"] += 1
            state["gei_buffer"] = gei.copy() if gei is not None else None

            for i, img in enumerate([frame, silhouette, gei]):
                _, buffer = cv2.imencode(".jpg", img)
                img_encoded = base64.b64encode(buffer).decode("utf-8")
                socketio.emit(f"frame{i}", img_encoded)

            cv2.waitKey(1)
            time.sleep(0.03)

    finally:
        cam.stop()
        cv2.destroyAllWindows()


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("reset")
def handle_reset():
    state["reset_requested"] = True


@socketio.on("save")
def handle_save_gei(data):
    label = data.get("label")
    gei_base64 = data.get("gei")

    if not label or not gei_base64:
        print("Invalid GEI save request.")
        return

    try:
        payload = {"label": label, "gei": gei_base64}

        response = requests.post("http://localhost:5001/api/register", json=payload)
        print(response.content)

    except Exception as e:
        print(f"Failed to save GEI: {e}")

    state["reset_requested"] = True


if __name__ == "__main__":
    threading.Thread(target=gei_save_thread, daemon=True).start()

    threading.Thread(target=main, daemon=True).start()

    socketio.run(app, host="0.0.0.0", port=5000)
