import requests
from flask import Blueprint, render_template
from state import state  # Import the shared state

routes = Blueprint("routes", __name__)


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
            payload = {"label": label, "gei": gei_base64}

            response = requests.post("http://localhost:5001/api/register", json=payload)
            print(response.content)

        except Exception as e:
            print(f"Failed to save GEI: {e}")

        state["reset_requested"] = True


@routes.route("/")
def index():
    return render_template("index.html")
