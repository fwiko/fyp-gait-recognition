import pickle

import cv2
import numpy as np
from models import GaitModel, GaitSample, db
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model


# Load pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights="imagenet", include_top=False)
# Create a model that outputs the features from the last convolutional layer
feature_extractor = Model(
    inputs=base_model.input, outputs=base_model.get_layer("top_conv").output
)


def preprocess_gei(gei):
    # Resize to EfficientNetB0 input size (224x224)
    resized = cv2.resize(gei, (224, 224))
    # Convert to 3 channels (EfficientNetB0 expects RGB)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    # Preprocess for EfficientNetB0
    preprocessed = preprocess_input(rgb)
    return preprocessed


def extract_features(gei):
    if gei is None:
        return None

    # Preprocess the GEI image to the required input size and format for the model
    preprocessed = preprocess_gei(gei)

    # Add batch dimension required by the model
    batch = np.expand_dims(preprocessed, axis=0)

    # Extract features using CNN
    features = feature_extractor.predict(batch, verbose=0)

    # Flatten the features
    features = features.flatten()

    return features


def update_model():
    try:
        # Get all gait samples
        gait_samples = GaitSample.query.all()
        if not gait_samples:
            return None

        # Extract features from all samples
        features = np.array([pickle.loads(sample.features) for sample in gait_samples])

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Perform PCA
        n_components = min(128, len(features) - 1)
        pca = PCA(n_components=n_components)
        pca.fit(features_scaled)

        # Create or update the model
        gait_model = GaitModel.query.first()
        if not gait_model:
            gait_model = GaitModel()

        gait_model.pca_components = pickle.dumps(pca.components_)
        gait_model.mean_vector = pickle.dumps(scaler.mean_)
        gait_model.n_components = n_components

        db.session.add(gait_model)
        db.session.commit()

        return gait_model

    except Exception as e:
        db.session.rollback()
        print(f"Error updating gait model: {e.__module__}")
        return None


def classify(gei, confidence_threshold=75.0):
    # Extract features using CNN
    features = extract_features(gei)
    if features is None or features.size == 0:
        return None, 0

    # Get the current model
    gait_model = GaitModel.query.first()
    if not gait_model:
        return None, 0

    # Load model components
    pca_components = pickle.loads(gait_model.pca_components)
    mean_vector = pickle.loads(gait_model.mean_vector)

    # Standardize and project features
    features_scaled = (features - mean_vector) / np.sqrt(np.var(features))
    projected_features = np.dot(features_scaled, pca_components.T)

    # Get all gait samples
    gait_samples = GaitSample.query.all()
    if not gait_samples:
        return None, 0

    # Calculate similarities
    similarities = []

    for sample in gait_samples: # For each registered gait sample
        sample_features = pickle.loads(sample.features) # load the already extracted features
        sample_features_scaled = (sample_features - mean_vector) / np.sqrt(
            np.var(sample_features)
        ) # Standardise the sample features and project them to the PCA space
        sample_projected = np.dot(sample_features_scaled, pca_components.T)

        similarities.append(
            (
                cosine_similarity(
                    projected_features.reshape(1, -1),
                    sample_projected.reshape(1, -1),
                )[0][0],
                sample.identity.label if sample.identity else None,
            ) # Calculate the cosine similarity between the projected features and the sample projected features
        )

    # If no valid comparisons
    if not similarities:
        return None, 0

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Get the best match
    best_similarity, prediction = similarities[0]

    # Convert similarity to confidence percentage (0-100)
    confidence = round(best_similarity * 100, 2)

    return prediction, confidence
