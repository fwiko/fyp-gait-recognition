from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Identity(db.Model):
    __tablename__ = "Identities"
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
    __tablename__ = "GaitCycles"
    id = db.Column(db.Integer, primary_key=True)
    identity_id = db.Column(db.Integer, db.ForeignKey("Identities.id"), nullable=False)
    gei_image = db.Column(db.Text, nullable=False)
    features = db.Column(db.LargeBinary, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)


class AccessRule(db.Model):
    __tablename__ = "AccessRules"
    identity_id = db.Column(
        db.Integer, db.ForeignKey("Identities.id"), primary_key=True
    )
    rule = db.Column(db.Boolean, nullable=False, default=False)


class GaitModel(db.Model):
    __tablename__ = "GaitModels"
    id = db.Column(db.Integer, primary_key=True)
    pca_components = db.Column(db.LargeBinary, nullable=False)
    mean_vector = db.Column(db.LargeBinary, nullable=False)
    n_components = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)


class ActivityLog(db.Model):
    __tablename__ = "ActivityLogs"
    id = db.Column(db.Integer, primary_key=True)
    activity_type = db.Column(db.String(20), nullable=False)
    details = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
