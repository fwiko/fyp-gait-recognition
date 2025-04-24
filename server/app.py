import os

from flask import Flask
from flask_socketio import SocketIO
from models import db
from routes import register_socket_events, routes

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(
    os.path.dirname(__file__), "gait_recognition.db"
) # set the database URI to the path of the gait_recognition.db file

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading") # create a socketio instance

db.init_app(app) # initialise the database

register_socket_events(socketio) # register the socket events

with app.app_context():
    db.create_all() # create all the tables in the database (if they don't already exist)

app.register_blueprint(routes) # register the routes

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
