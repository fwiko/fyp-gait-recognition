import numpy as np
import pickle
import cv2
import base64
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from models import GaitSample, GaitModel, db

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# Load pre-trained VGG16 model
base_model = VGG16(weights="imagenet", include_top=False)
# Create a model that outputs the features from the last convolutional layer
feature_extractor = Model(
    inputs=base_model.input, outputs=base_model.get_layer("block5_conv3").output
)


def preprocess_gei(gei):
    """Preprocess GEI image for CNN input"""
    # Resize to VGG16 input size (224x224)
    resized = cv2.resize(gei, (224, 224))
    # Convert to 3 channels (VGG16 expects RGB)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    # Preprocess for VGG16
    preprocessed = preprocess_input(rgb)
    return preprocessed


def extract_features(gei):
    """Extract features from GEI image using CNN"""
    if gei is None:
        return None

    # Preprocess the GEI image
    preprocessed = preprocess_gei(gei)

    # Add batch dimension
    batch = np.expand_dims(preprocessed, axis=0)

    # Extract features using CNN
    features = feature_extractor.predict(batch, verbose=0)

    # Flatten the features
    features = features.flatten()

    return features


def update_model():
    """Update the PCA model with all available gait samples"""
    try:
        # Get all gait samples
        gait_samples = GaitSample.query.all()
        if not gait_samples:
            return None

        # Extract features from all samples
        features_list = []
        labels = []
        for sample in gait_samples:
            # Decode the base64 image
            gei_bytes = base64.b64decode(sample.gei_image)
            nparr = np.frombuffer(gei_bytes, np.uint8)
            gei = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            # Extract features using CNN
            features = extract_features(gei)
            features_list.append(features)
            labels.append(sample.identity.label if sample.identity else None)

        # Convert to numpy array
        X = np.array(features_list)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform PCA
        n_components = min(
            128, len(features_list) - 1
        )  # Use more components for CNN features
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

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
        print(f"Error updating gait model: {str(e)}")
        return None


def classify(gei, confidence_threshold=75.0):
    """Classify GEI using CNN features and cosine similarity"""
    # Extract features using CNN
    features = extract_features(gei)
    if features is None or features.size == 0:
        return "Unknown", 0

    # Get the current model
    gait_model = GaitModel.query.first()
    if not gait_model:
        return "Unknown", 0

    # Load model components
    pca_components = pickle.loads(gait_model.pca_components)
    mean_vector = pickle.loads(gait_model.mean_vector)

    # Standardize and project features
    features_scaled = (features - mean_vector) / np.sqrt(np.var(features))
    projected_features = np.dot(features_scaled, pca_components.T)

    # Get all gait samples
    gait_samples = GaitSample.query.all()
    if not gait_samples:
        return "Unknown", 0

    # Calculate similarities
    similarities = []
    for sample in gait_samples:
        # Decode the base64 image
        gei_bytes = base64.b64decode(sample.gei_image)
        nparr = np.frombuffer(gei_bytes, np.uint8)
        sample_gei = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Extract features using CNN
        sample_features = extract_features(sample_gei)
        sample_features_scaled = (sample_features - mean_vector) / np.sqrt(
            np.var(sample_features)
        )
        sample_projected = np.dot(sample_features_scaled, pca_components.T)

        # Calculate cosine similarity
        similarity = cosine_similarity(
            projected_features.reshape(1, -1), sample_projected.reshape(1, -1)
        )[0][0]

        # Assuming `identity` relationship is loaded
        label = sample.identity.label if sample.identity else None
        similarities.append((similarity, label))

    # If no valid comparisons
    if not similarities:
        return "Unknown", 0

    # Sort by similarity
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Get the best match
    best_similarity, prediction = similarities[0]

    # Convert similarity to confidence (0-100)
    confidence = round(best_similarity * 100, 2)

    # Apply confidence threshold
    if confidence < confidence_threshold:
        return "Unknown", confidence

    return prediction, confidence
