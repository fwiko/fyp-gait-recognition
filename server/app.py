import os

from flask import Flask
from flask_socketio import SocketIO
from models import db
from routes import register_socket_events, routes

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(
    os.path.dirname(__file__), "gait_recognition.db"
)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

db.init_app(app)

register_socket_events(socketio)

with app.app_context():
    db.create_all()

app.register_blueprint(routes)
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
