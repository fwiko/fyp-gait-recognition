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
    """Extract gait-specific features from GEI image"""
    if gei is None:
        return None

    # Extract horizontal and vertical projections
    h_proj = np.sum(gei, axis=1)
    v_proj = np.sum(gei, axis=0)

    # Extract center of mass
    y_indices, x_indices = np.indices(gei.shape)
    total_mass = np.sum(gei)
    if total_mass > 0:
        com_x = np.sum(x_indices * gei) / total_mass
        com_y = np.sum(y_indices * gei) / total_mass
    else:
        com_x, com_y = gei.shape[1] // 2, gei.shape[0] // 2

    # Extract width and height at different intensity levels
    intensity_levels = [0.2, 0.4, 0.6, 0.8]
    width_heights = []
    for level in intensity_levels:
        mask = gei >= level
        if np.any(mask):
            y_coords, x_coords = np.where(mask)
            width = np.max(x_coords) - np.min(x_coords) if len(x_coords) > 0 else 0
            height = np.max(y_coords) - np.min(y_coords) if len(y_coords) > 0 else 0
            width_heights.extend([width, height])
        else:
            width_heights.extend([0, 0])

    # Extract gait-specific features
    # 1. Step length estimation (horizontal distance between feet)
    left_side = np.sum(gei[:, : gei.shape[1] // 2])
    right_side = np.sum(gei[:, gei.shape[1] // 2 :])
    step_length = abs(left_side - right_side) / total_mass if total_mass > 0 else 0

    # 2. Stride length (vertical distance between steps)
    upper_half = np.sum(gei[: gei.shape[0] // 2, :])
    lower_half = np.sum(gei[gei.shape[0] // 2 :, :])
    stride_length = abs(upper_half - lower_half) / total_mass if total_mass > 0 else 0

    # 3. Gait symmetry features
    symmetry_x = (
        np.sum(np.abs(gei - np.fliplr(gei))) / total_mass if total_mass > 0 else 0
    )
    symmetry_y = (
        np.sum(np.abs(gei - np.flipud(gei))) / total_mass if total_mass > 0 else 0
    )

    # 4. Gait energy distribution
    energy_dist = np.histogram(gei.flatten(), bins=10, range=(0, 1))[0]
    energy_dist = (
        energy_dist / np.sum(energy_dist) if np.sum(energy_dist) > 0 else energy_dist
    )

    # Combine all features
    features = np.concatenate(
        [
            h_proj,  # Horizontal projection
            v_proj,  # Vertical projection
            [com_x, com_y],  # Center of mass
            width_heights,  # Width and height at different intensities
            [step_length, stride_length],  # Step and stride length
            [symmetry_x, symmetry_y],  # Symmetry features
            energy_dist,  # Energy distribution
        ]
    )

    return features


def knn_classify(gei, confidence_threshold=75.0, k=5):
    """Classify GEI using KNN with gait-specific features"""
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

        # Feature-specific weights
        weights = np.ones_like(features)

        # Higher weights for gait-specific features
        proj_len = len(features) - 14  # Length of projections
        weights[:proj_len] = 2.0  # Double weight for projections
        weights[proj_len : proj_len + 2] = 1.5  # Center of mass
        weights[proj_len + 2 : proj_len + 10] = 1.2  # Width/height features
        weights[proj_len + 10 : proj_len + 12] = 1.8  # Step/stride length
        weights[proj_len + 12 : proj_len + 14] = 1.5  # Symmetry features
        weights[proj_len + 14 :] = 1.0  # Energy distribution

        # Normalize weights
        weights = weights / np.sum(weights)

        # Calculate weighted Euclidean distance
        diff = features - cycle_features
        weighted_diff = diff * weights
        dist = np.sqrt(np.sum(weighted_diff**2))

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
    k = min(k, len(distances))
    nearest = distances[:k]

    # Count labels with distance-based weights
    label_counts = {}
    total_weight = 0
    for dist, label in nearest:
        if label:
            # Exponential weight decay based on distance
            weight = np.exp(-dist / mean_dist) if mean_dist > 0 else 1.0
            label_counts[label] = label_counts.get(label, 0) + weight
            total_weight += weight

    if not label_counts:
        return "Unknown", 0

    # Find most common weighted label
    prediction = max(label_counts.items(), key=lambda x: x[1])[0]
    prediction_weight = label_counts[prediction]

    # Calculate confidence based on weighted votes and distance
    vote_confidence = (prediction_weight / total_weight) * 100

    # Get matched distances
    matched_dists = [d for d, l in nearest if l == prediction]
    avg_match_dist = np.mean(matched_dists)

    # Calculate Z-score
    if std_dist > 0:
        z_score = (mean_dist - avg_match_dist) / std_dist
    else:
        z_score = 0

    # Convert Z-score to confidence using sigmoid
    distance_confidence = round(100 / (1 + np.exp(-z_score)), 2)

    # Final confidence is a weighted average with more emphasis on distance
    final_confidence = round(0.7 * distance_confidence + 0.3 * vote_confidence, 2)

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
    gait_cycles = identity.gait_cycles
    return render_template("identity.html", identity=identity, gait_cycles=gait_cycles)


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


@app.route("/access_rules", methods=["GET"])
def access_rules():
    identities = Identity.query.all()
    return render_template("access_rules.html", identities=identities)


@app.route("/stats")
def stats():
    total_identities = Identity.query.count()
    total_gait_cycles = GaitCycle.query.count()
    latest_samples = (
        db.session.query(
            Identity.label, db.func.max(GaitCycle.created_at).label("latest_sample")
        )
        .join(GaitCycle)
        .group_by(Identity.id)
        .limit(5)
        .all()
    )

    return render_template(
        "stats.html",
        total_identities=total_identities,
        total_gait_cycles=total_gait_cycles,
        latest_samples=latest_samples,
    )


@app.route("/api/access_rule/<int:identity_id>", methods=["PUT"])
def update_access_rule(identity_id):
    """API endpoint to update access rule for an identity"""
    try:
        data = request.json
        if not data or "allow_access" not in data:
            return jsonify({"success": False, "error": "Missing required data"}), 400

        identity = Identity.query.get_or_404(identity_id)
        allow_access = data["allow_access"]

        if identity.access_rule:
            identity.access_rule.rule = allow_access
        else:
            new_rule = AccessRule(identity_id=identity.id, rule=allow_access)
            db.session.add(new_rule)

        db.session.commit()
        return jsonify({"success": True})

    except Exception as e:
        db.session.rollback()
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
