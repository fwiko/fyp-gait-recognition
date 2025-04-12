import os
import cv2
import base64
import threading
import time
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
}

# Define save directory for GEIs
SAVE_DIR = "saved_geis"

# Save GEI every 150 frames
FRAME_INTERVAL = 150


def save_gei_to_disk(gei):
    """Save the GEI image to disk."""
    try:
        # Ensure the save directory exists
        os.makedirs(SAVE_DIR, exist_ok=True)

        # Create a filename with current timestamp
        filename = f"gei_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)

        # Save the GEI image as a JPG file
        _, buffer = cv2.imencode(".jpg", gei)
        img_data = base64.b64encode(buffer).decode("utf-8")

        # Save the image to the disk
        with open(save_path, "wb") as f:
            f.write(base64.b64decode(img_data))

        print(f"Saved GEI at {save_path}")
    except Exception as e:
        print(f"Failed to save GEI: {e}")


def gei_save_thread():
    """This thread will save the GEI every 150 frames."""
    while True:
        if state["frame_counter"] >= FRAME_INTERVAL:
            print("Saving GEI to disk...")
            if state["gei_buffer"] is not None:
                # save_gei_to_disk(state["gei_buffer"])
                pass
            # Reset frame counter after saving
            state["frame_counter"] = 0

        # Sleep to avoid high CPU usage
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
                if processor.initialize_background(frame):
                    state["background_init"] = True
                    print("Background initialized!")
                continue

            sil, gei = processor.process_frame(frame)

            # Increment frame counter
            state["frame_counter"] += 1

            # Store the GEI for saving later
            state["gei_buffer"] = gei

            # Send frames to frontend
            for i, img in enumerate([frame, sil, gei]):
                _, buffer = cv2.imencode(".jpg", img)
                img_encoded = base64.b64encode(buffer).decode("utf-8")
                socketio.emit(f"frame{i}", img_encoded)

            cv2.waitKey(1)

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
        img_data = base64.b64decode(gei_base64)
        filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)

        os.makedirs(SAVE_DIR, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(img_data)

        print(f"Saved GEI for '{label}' at {save_path}")
    except Exception as e:
        print(f"Failed to save GEI: {e}")

    state["reset_requested"] = True


if __name__ == "__main__":
    threading.Thread(target=gei_save_thread, daemon=True).start()

    threading.Thread(target=main, daemon=True).start()

    socketio.run(app, host="0.0.0.0", port=5000)
