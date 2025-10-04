import base64
import threading
import time

import cv2
import numpy as np
from camera import Camera
from flask import Flask
from flask_socketio import SocketIO
from gait import GaitProcessor
from routes import register_socket_events, routes
from server_api import ServerAPIClient
from state import state

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
api_client = ServerAPIClient()


FRAME_INTERVAL = 30  # Number of frames between GEI classification attempts


# Check if two GEI NumPy arrays are different
def is_different(gei_a, gei_b, threshold=15.0):
    if gei_a is None or gei_b is None:
        return True

    if gei_a.shape != gei_b.shape:
        return True

    diff = np.linalg.norm(gei_a.astype("float32") - gei_b.astype("float32"))
    return diff > threshold


# Thread to perform GEI classification in a loop
def classification_thread():
    while True:
        if state["frame_counter"] >= FRAME_INTERVAL:
            current_gei = state["gei_buffer"]
            last_saved_gei = state["last_saved_gei"]

            if current_gei is not None:
                if is_different(current_gei, last_saved_gei):

                    # Encode the GEI as a base64 string
                    _, buffer = cv2.imencode(".jpg", current_gei)
                    gei_base64 = base64.b64encode(buffer).decode("utf-8")

                    state["last_saved_gei"] = current_gei

                    # Classify the GEI
                    response, message = api_client.classify_gei(gei_base64)

                    if not response:
                        print(f"Failed to classify GEI: {message}")
                    else:
                        print(response)
                        socketio.emit("status", response)

            state["frame_counter"] = 0  # Reset the frame counter

        time.sleep(1)


def main():
    # Initialise camera (ID: 1 for the external webcam)
    cam = Camera(camera_id=0)
    cam.start()

    # Initialise the gait processor with a buffer size of 30 frames (number of frames contributing to a GEI)
    processor = GaitProcessor(buffer_size=30)

    try:
        while True:
            # Get all existing WebSocket connections
            connected_clients = [
                sid
                for sid, environ in socketio.server.environ.items()
                if environ.get("HTTP_REFERER", "").endswith("/")
            ]

            # If there are no connected clients, skip loop iteration
            if len(socketio.server.eio.sockets) == 0 or not connected_clients:
                time.sleep(0.1)
                continue

            # Get the latest frame from the camera
            frame = cam.get_frame()
            if frame is None:
                continue

            # If a reset has been requested, reinitialise the gait processor's background model
            if state["reset_requested"]:
                state["background_init"] = False
                processor.reset_background()
                state["reset_requested"] = False

            # If the gait processor's background model has not been initialised, initialise it
            if not state["background_init"]:
                if processor.initialise_backround(frame):
                    state["background_init"] = True
                    print("Background initialized!")
                continue

            # Process the frame to get the silhouette and most recent (if available given required buffer size) GEI
            silhouette, gei = processor.process_frame(frame)

            state["frame_counter"] += 1
            state["gei_buffer"] = gei.copy() if gei is not None else None

            # Transmit the initial frame, silhouette and GEI to the client (front-end) via WebSocket
            for i, img in enumerate([frame, silhouette, gei]):
                _, buffer = cv2.imencode(".jpg", img)
                img_encoded = base64.b64encode(buffer).decode("utf-8")
                socketio.emit(f"frame{i}", img_encoded)

            cv2.waitKey(1)
            time.sleep(0.03)

    finally:
        cam.stop()
        cv2.destroyAllWindows()


register_socket_events(socketio)  # Register WebSocket events
app.register_blueprint(routes)  # Register routes

if __name__ == "__main__":
    threading.Thread(
        target=classification_thread, daemon=True
    ).start()  # Start the classification thread

    threading.Thread(target=main, daemon=True).start()

    socketio.run(app, host="0.0.0.0", port=5000)
