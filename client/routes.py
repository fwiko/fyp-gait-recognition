import base64

from flask import Blueprint, render_template
from server_api import ServerAPIClient  # Import the API client
from state import state  # Import the shared state

routes = Blueprint("routes", __name__)
api_client = ServerAPIClient()  # Initialize the API client


def register_socket_events(socketio):
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
            # Decode the base64 image to bytes
            gei_bytes = base64.b64decode(gei_base64)

            # Use the API client to register the GEI
            success, error = api_client.register_gei(gei_bytes, label)

            if error:
                print(f"Failed to save GEI: {error}")
            else:
                print("GEI saved successfully")

        except Exception as e:
            print(f"Failed to save GEI: {e}")

        state["reset_requested"] = True


@routes.route("/")
def index():
    return render_template("index.html")
