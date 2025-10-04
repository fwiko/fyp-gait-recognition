import base64
import pickle
from datetime import datetime

import cv2
import numpy as np
from flask import Blueprint, jsonify, render_template, request
from flask_socketio import emit
from gait import classify, extract_features, update_model
from models import AccessRule, ActivityLog, GaitSample, Identity, db

routes = Blueprint("routes", __name__)


# Socket.IO event handlers
def register_socket_events(socketio):
    @socketio.on("update-access-rule")
    def handle_access_rule_update(data):
        try:
            identity_id = data.get("identity_id")
            allow_access = data.get("allow_access")

            if not identity_id or allow_access is None:
                emit(
                    "access_rule_updated",
                    {"success": False, "error": "Missing required data"},
                )
                return

            identity = Identity.query.get_or_404(identity_id)
            old_state = identity.access_rule.rule if identity.access_rule else False
            identity_label = identity.label

            if identity.access_rule:
                identity.access_rule.rule = allow_access
            else:
                new_rule = AccessRule(identity_id=identity.id, rule=allow_access)
                db.session.add(new_rule)

            # Log the access change activity
            activity_log = ActivityLog(
                activity_type="access_change",
                details=str(
                    {
                        "label": identity_label,
                        "old_state": old_state,
                        "new_state": allow_access,
                    }
                ),
                created_at=datetime.now(),
            )
            db.session.add(activity_log)

            db.session.commit()

            # Emit success response
            emit("access_rule_updated", {"success": True, "allow_access": allow_access})

        except Exception as e:
            db.session.rollback()
            emit("access_rule_updated", {"success": False, "error": str(e)})

    @socketio.on("delete-identity")
    def handle_delete_identity(data):
        try:
            identity_id = data.get("identity_id")
            if not identity_id:
                emit(
                    "identity_deleted",
                    {"success": False, "error": "Missing identity ID"},
                )
                return

            identity = Identity.query.get_or_404(identity_id)

            activity_log = ActivityLog(
                activity_type="delete",
                details=str({"label": identity.label}),
                created_at=datetime.now(),
            )
            db.session.add(activity_log)

            db.session.delete(identity)
            db.session.commit()

            update_model()

            emit("identity_deleted", {"success": True})

        except Exception as e:
            db.session.rollback()
            emit("identity_deleted", {"success": False, "error": str(e)})

    @socketio.on("delete-gait-sample")
    def handle_delete_gait_sample(data):
        try:
            sample_id = data.get("sample_id")
            if not sample_id:
                emit(
                    "gait_sample_deleted",
                    {"success": False, "error": "Missing sample ID"},
                )
                return

            sample = GaitSample.query.get_or_404(sample_id)
            identity_label = sample.identity.label  # Store label before deletion

            # Log the deletion activity
            activity_log = ActivityLog(
                activity_type="delete",
                details=str({"label": identity_label, "sample_id": sample_id}),
                created_at=datetime.now(),
            )
            db.session.add(activity_log)

            # Delete the gait sample
            db.session.delete(sample)
            db.session.commit()

            # Update the PCA model after deleting a gait sample
            update_model()

            emit("gait_sample_deleted", {"success": True, "sample_id": sample_id})

        except Exception as e:
            db.session.rollback()
            emit("gait_sample_deleted", {"success": False, "error": str(e)})


@routes.route("/")
def index() -> str:
    identities = Identity.query.all()
    return render_template("index.html", identities=identities)


@routes.route("/identity/<int:identity_id>")
def view_identity(identity_id) -> str:
    identity = Identity.query.get_or_404(identity_id)
    gait_samples = identity.gait_samples
    return render_template(
        "identity.html", identity=identity, gait_samples=gait_samples
    )


@routes.route("/stats")
def stats() -> str:
    total_identities = Identity.query.count()
    total_gait_samples = GaitSample.query.count()

    recent_activity = ActivityLog.query.order_by(ActivityLog.created_at.desc()).all()

    formatted_logs = []
    for log in recent_activity:
        timestamp = log.created_at.strftime("%Y-%m-%d %H:%M:%S")
        details = eval(log.details) if log.details else {}

        if log.activity_type == "register":
            label = details.get("label", "Unknown")
            message = f"Registered new gait sample for {label}"
        elif log.activity_type == "delete":
            label = details.get("label", "Unknown")
            if details.get("type") == "identity":
                message = f"Deleted identity: {label}"
            else:
                message = f"Deleted gait sample for {label}"
        elif log.activity_type == "access_change":
            label = details.get("label", "Unknown")
            new_state = "granted" if details.get("new_state") else "denied"
            message = f"Access {new_state} for {label}"
        elif log.activity_type == "identify":
            label = details.get("label", "Unknown")
            confidence = details.get("confidence", 0)
            message = f"Identified as {label} (confidence: {confidence}%)"
        else:
            message = f"Unknown activity"

        formatted_logs.append(
            {"timestamp": timestamp, "message": message, "type": log.activity_type}
        )

    return render_template(
        "stats.html",
        total_identities=total_identities,
        total_gait_samples=total_gait_samples,
        activity_logs=formatted_logs,
    )


# API Endpoints
@routes.route("/api/register", methods=["POST"])
def register_gei() -> tuple[jsonify, int]:
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

        # Create new gait sample
        new_sample = GaitSample(
            identity_id=identity.id,
            gei_image=base64_gei,
            features=serialized_features,
            created_at=datetime.now(),
        )

        db.session.add(new_sample)

        # Log the registration activity
        activity_log = ActivityLog(
            activity_type="register",
            details=str({"label": label}),
            created_at=datetime.now(),
        )
        db.session.add(activity_log)

        db.session.commit()

        # Update the PCA model after adding new data
        update_model()

        return jsonify({"success": True, "identity_id": identity.id})

    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@routes.route("/api/classify", methods=["POST"])
def classify_gei() -> tuple[jsonify, int]:
    """API endpoint to classify a GEI image using PCA and cosine similarity"""
    data = request.json

    if not data or "gei" not in data:
        return jsonify({"success": False, "error": "Missing GEI data"}), 400

    try:
        base64_gei = data["gei"]

        gei = base64.b64decode(base64_gei)
        nparr = np.frombuffer(gei, np.uint8)
        gei = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        prediction, confidence = classify(gei)

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

        # Log the identification activity
        activity_log = ActivityLog(
            activity_type="identify",
            details=str(
                {
                    "confidence": confidence,
                    "label": prediction,
                }
            ),
            created_at=datetime.now(),
        )
        db.session.add(activity_log)
        db.session.commit()

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
