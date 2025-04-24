import base64

from flask import Blueprint, jsonify, render_template, request
from server_api import ServerAPIClient
from state import state

routes = Blueprint("routes", __name__)
api_client = ServerAPIClient()


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
            gei_bytes = base64.b64decode(gei_base64)

            _, message = api_client.register_gei(gei_bytes, label) # Submit a registration request to the Gait Recognition Server

            if message:
                print(f"Failed to save GEI: {message}")

        except Exception as e:
            print(f"Failed to save GEI: {e}")

        state["reset_requested"] = True


@routes.route("/")
def index():
    return render_template("index.html")


@routes.route("/manual")
def manual():
    return render_template("manual.html")


@routes.route("/api/classify", methods=["POST"])
def classify_gei():
    try:
        data = request.get_json()

        if not data or "gei" not in data:
            return jsonify({"error": "No GEI image provided"}), 400

        response, message = api_client.classify_gei(data["gei"]) # Submit a classification request to the Gait Recognition Server

        if not response:
            return jsonify({"error": message}), 500

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
