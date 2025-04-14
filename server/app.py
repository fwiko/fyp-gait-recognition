import os
import base64
import io
import cv2
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///gait_recognition.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Initialize SQLAlchemy
db = SQLAlchemy(app)


# Define database models
class Identity(db.Model):
    __tablename__ = "identities"
    id = db.Column(db.Integer, primary_key=True)
    label = db.Column(db.String(50), nullable=False)
    gait_cycles = db.relationship(
        "GaitCycle", backref="identity", lazy=True, cascade="all, delete-orphan"
    )
    access_rule = db.relationship(
        "AccessRule",
        backref="identity",
        lazy=True,
        uselist=False,
        cascade="all, delete-orphan",
    )


class GaitCycle(db.Model):
    __tablename__ = "gaitcycles"
    id = db.Column(db.Integer, primary_key=True)
    identity_id = db.Column(db.Integer, db.ForeignKey("identities.id"), nullable=False)
    gei_image = db.Column(db.Text, nullable=False)  # Base64 encoded image
    knn_features = db.Column(db.LargeBinary, nullable=False)  # Serialized features
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class AccessRule(db.Model):
    __tablename__ = "accessrules"
    identity_id = db.Column(
        db.Integer, db.ForeignKey("identities.id"), primary_key=True
    )
    rule = db.Column(db.Boolean, nullable=False, default=False)


# Create database tables
with app.app_context():
    db.create_all()


def extract_features(gei):
    if gei is None:
        return None

    features = gei.flatten()

    return features


def knn_classify(gei, confidence_threshold=75.0):
    features = extract_features(gei)
    if features is None or features.size == 0:
        return "Unknown", 0

    gait_cycles = GaitCycle.query.all()
    if not gait_cycles:
        return "Unknown", 0

    # Calculate distances to all samples
    distances = []
    for cycle in gait_cycles:
        cycle_features = pickle.loads(cycle.knn_features)

        # Ensure features are the same size
        if cycle_features.size != features.size:
            continue

        # Euclidean distance
        dist = np.linalg.norm(features - cycle_features)

        # Assuming `identity` relationship is loaded
        label = cycle.identity.label if cycle.identity else None
        distances.append((dist, label))

    # If no valid comparisons
    if not distances:
        return "Unknown", 0

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Calculate distance statistics for all samples
    all_distances = [d for d, _ in distances]
    mean_dist = np.mean(all_distances)
    std_dist = np.std(all_distances)

    # Get k nearest neighbors
    k = min(3, len(distances))
    nearest = distances[:k]

    # Count labels
    label_counts = {}
    for dist, label in nearest:
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1

    if not label_counts:
        return "Unknown", 0

    # Find most common label
    prediction = max(label_counts.items(), key=lambda x: x[1])[0]

    # Get matched distances
    matched_dists = [d for d, l in nearest if l == prediction]
    avg_match_dist = np.mean(matched_dists)

    # Calculate Z-score (how many standard deviations from the mean)
    # Smaller distances = larger z-scores = better matches
    if std_dist > 0:
        z_score = (mean_dist - avg_match_dist) / std_dist
    else:
        z_score = 0

    # Convert Z-score to confidence (sigmoid function)
    # This maps z-scores to a range of 0-100
    # Z-score of 0 gives 50% confidence
    # Positive z-scores (better than average) give >50%
    # Negative z-scores (worse than average) give <50%
    confidence = round(100 / (1 + np.exp(-z_score)), 2)

    # Vote confidence component (how many neighbors voted for this label)
    vote_confidence = (label_counts[prediction] / k) * 100

    # Final confidence is a weighted average
    final_confidence = round(0.7 * confidence + 0.3 * vote_confidence, 2)

    # Apply confidence threshold
    if final_confidence < confidence_threshold:
        return "Unknown", final_confidence

    return prediction, final_confidence


# Routes
@app.route("/")
def index():
    identities = Identity.query.all()
    return render_template("index.html", identities=identities)


@app.route("/identity/<int:identity_id>")
def view_identity(identity_id):
    identity = Identity.query.get_or_404(identity_id)
    return render_template("identity.html", identity=identity)


@app.route("/identity/<int:identity_id>/update", methods=["POST"])
def update_identity(identity_id):
    identity = Identity.query.get_or_404(identity_id)

    # Update identity label
    identity.label = request.form.get("label")

    # Update access rule
    access = request.form.get("access") == "true"
    if identity.access_rule:
        identity.access_rule.rule = access
    else:
        new_rule = AccessRule(identity_id=identity.id, rule=access)
        db.session.add(new_rule)

    db.session.commit()
    return redirect(url_for("view_identity", identity_id=identity_id))


@app.route("/identity/<int:identity_id>/delete", methods=["POST"])
def delete_identity(identity_id):
    identity = Identity.query.get_or_404(identity_id)
    db.session.delete(identity)
    db.session.commit()
    return redirect(url_for("index"))


# API Endpoints
@app.route("/api/register", methods=["POST"])
def register_gei():
    """API endpoint to register a new GEI image"""
    data = request.json

    if not data or "label" not in data or "gei" not in data:
        return jsonify({"success": False, "error": "Missing required data"}), 400

    try:
        label = data["label"]
        base64_gei = data["gei"]

        # Check if identity exists
        identity = Identity.query.filter_by(label=label).first()

        # If not, create new identity with default access rule (denied)
        if not identity:
            identity = Identity(label=label)
            db.session.add(identity)
            db.session.flush()

            access_rule = AccessRule(identity_id=identity.id, rule=False)
            db.session.add(access_rule)

        gei = base64.b64decode(base64_gei)
        nparr = np.frombuffer(gei, np.uint8)
        gei = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        features = extract_features(gei)
        if features is None:
            return (
                jsonify({"success": False, "error": "Failed to extract features"}),
                400,
            )

        serialized_features = pickle.dumps(features)

        # Create new gait cycle
        new_cycle = GaitCycle(
            identity_id=identity.id,
            gei_image=base64_gei,
            knn_features=serialized_features,
            created_at=datetime.now(),
        )

        db.session.add(new_cycle)
        db.session.commit()

        return jsonify({"success": True, "identity_id": identity.id})

    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/verify", methods=["POST"])
def verify_gei():
    """API endpoint to verify a GEI image using KNN"""
    data = request.json

    if not data or "gei" not in data:
        return jsonify({"success": False, "error": "Missing GEI data"}), 400

    try:
        base64_gei = data["gei"]

        gei = base64.b64decode(base64_gei)
        nparr = np.frombuffer(gei, np.uint8)
        gei = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        features = extract_features(gei)

        if features is None:
            return (
                jsonify({"success": False, "error": "Failed to extract features"}),
                400,
            )

        prediction, confidence = knn_classify(gei)

        # If unknown, return early
        if prediction == "Unknown":
            return jsonify(
                {
                    "success": True,
                    "person": "Unknown",
                    "confidence": float(confidence),
                    "access": False,
                }
            )

        identity = Identity.query.filter_by(label=prediction).first()

        if not identity:
            return jsonify({"success": False, "error": "Identity not found"}), 404

        access_granted = False
        if identity.access_rule:
            access_granted = identity.access_rule.rule

        return jsonify(
            {
                "success": True,
                "person": prediction,
                "confidence": float(confidence),
                "access": access_granted,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Socket.IO events
@socketio.on("connect")
def handle_connect():
    print("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
