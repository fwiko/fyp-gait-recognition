import base64
import json
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import requests
from camera import Camera
from flask import Flask, render_template
from flask_socketio import SocketIO
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

# Save GEI every 30 frames (increased from 10)
FRAME_INTERVAL = 30

# Maximum number of samples per identity
MAX_SAMPLES_PER_IDENTITY = 20


def is_different(gei_a, gei_b, threshold=15.0):  # Increased from 5.0
    if gei_a is None or gei_b is None:
        return True

    if gei_a.shape != gei_b.shape:
        return True

    # Calculate structural similarity index
    diff = np.linalg.norm(gei_a.astype("float32") - gei_b.astype("float32"))
    return diff > threshold


def save_gei(gei, label):
    """Save GEI image with label"""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Check if we've reached the maximum samples for this identity
    existing_samples = len(
        [f for f in os.listdir(SAVE_DIR) if f.startswith(f"{label}_")]
    )
    if existing_samples >= MAX_SAMPLES_PER_IDENTITY:
        print(f"Maximum samples reached for {label}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(filepath, gei)
    print(f"Saved GEI to {filepath}")


def gei_save_thread():
    while True:
        if state["frame_counter"] >= FRAME_INTERVAL:
            current_gei = state["gei_buffer"]
            last_saved_gei = state["last_saved_gei"]

            if current_gei is not None:
                # Check if GEI has changed since last save
                if is_different(current_gei, last_saved_gei):
                    print("GEI has changed. Classifying...")

                    _, buffer = cv2.imencode(".jpg", current_gei)
                    gei_encoded = base64.b64encode(buffer).decode("utf-8")

                    state["last_saved_gei"] = current_gei

                    try:
                        response = requests.post(
                            "http://localhost:5001/api/classify",
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

        time.sleep(1)


def main():
    cam = Camera(camera_id=1)
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
