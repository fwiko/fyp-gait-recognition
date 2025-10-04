import os
import cv2
import numpy as np
import random
from tqdm import tqdm


class GaitProcessor:
    NORMALISED_WIDTH = 88
    NORMALISED_HEIGHT = 128

    def __init__(self):
        self.aligned_buffer = []

    def process_silhouette_sequence(self, silhouette_sequence):
        """Process a sequence of silhouette images to create a GEI."""
        aligned_silhouettes = []

        for silhouette in silhouette_sequence:
            # Find bounding box
            contours, _ = cv2.findContours(
                silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Apply alignment
            aligned = self._align_silhouette(silhouette, (x, y, w, h))
            if aligned is not None:
                aligned_silhouettes.append(aligned)

        if not aligned_silhouettes:
            return None

        # Create GEI by averaging aligned silhouettes
        gei = np.mean(aligned_silhouettes, axis=0)
        return (gei * 255).astype(np.uint8)

    def _align_silhouette(self, silhouette, bounding_box):
        x, y, w, h = bounding_box

        person_region = silhouette[y : y + h, x : x + w]

        ty, tx = np.where(person_region > 0)

        if len(ty) == 0 or len(tx) == 0:
            return None

        sy, ey = ty.min(), ty.max() + 1
        sx, ex = tx.min(), tx.max() + 1

        silhouette_h = ey - sy
        silhouette_w = ex - sx

        if silhouette_h <= silhouette_w:
            return None

        cx = int(tx.mean())
        cenX = silhouette_h // 2
        start_w = (silhouette_h - silhouette_w) // 2

        if max(cx - sx, ex - cx) < cenX:
            start_w = cenX - (cx - sx)

        square_silhouette = np.zeros((silhouette_h, silhouette_h), np.uint8)
        square_silhouette[:, start_w : start_w + silhouette_w] = person_region[
            sy:ey, sx:ex
        ]

        resized_silhouette = cv2.resize(
            square_silhouette,
            (self.NORMALISED_HEIGHT, self.NORMALISED_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

        offsetX = 20
        normalised_silhouette = resized_silhouette[
            :, offsetX : offsetX + self.NORMALISED_WIDTH
        ]

        return normalised_silhouette.astype(float) / 255.0


def load_silhouette_sequence(sequence_path):
    """Load a sequence of silhouette images from a directory."""
    silhouettes = []
    image_files = sorted(
        [f for f in os.listdir(sequence_path) if f.endswith((".png", ".jpg", ".bmp"))]
    )

    for image_file in image_files:
        image_path = os.path.join(sequence_path, image_file)
        silhouette = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if silhouette is not None:
            silhouettes.append(silhouette)

    return silhouettes


def prepare_dataset(root_dir, output_dir, train_split=0.5):
    """
    Prepare GEI dataset from the silhouette sequences.
    Only uses conditions starting with 'nm' and '090' direction.
    For each subject, some 'nm' samples go to training and others to testing.

    Args:
        root_dir: Root directory of the dataset
        output_dir: Directory to save the processed GEIs
        train_split: Fraction of 'nm' samples per subject to use for training
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    # Get all subject IDs
    subject_ids = [
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    ]

    # Initialize counters
    total_train_samples = 0
    total_test_samples = 0
    subject_stats = {}

    # Initialize gait processor
    processor = GaitProcessor()

    # Process each subject
    for subject_id in tqdm(subject_ids, desc="Processing subjects"):
        subject_path = os.path.join(root_dir, subject_id)
        subject_stats[subject_id] = {"train": 0, "test": 0}

        # Get all conditions for this subject
        conditions = [
            d
            for d in os.listdir(subject_path)
            if os.path.isdir(os.path.join(subject_path, d))
        ]

        # Filter conditions starting with 'nm'
        nm_conditions = [c for c in conditions if c.startswith("nm")]

        if not nm_conditions:
            continue

        # Randomize and split the nm conditions for this subject
        random.shuffle(nm_conditions)
        split_idx = max(
            1, int(len(nm_conditions) * train_split)
        )  # Ensure at least 1 sample for training
        train_conditions = nm_conditions[:split_idx]
        test_conditions = nm_conditions[split_idx:]

        # Make sure both sets have at least one sample
        if not test_conditions and len(train_conditions) > 1:
            test_conditions = [train_conditions.pop()]

        for condition in nm_conditions:
            condition_path = os.path.join(subject_path, condition)

            # Check for '090' direction
            directions = [
                d
                for d in os.listdir(condition_path)
                if os.path.isdir(os.path.join(condition_path, d))
            ]

            if "090" in directions:
                sequence_path = os.path.join(condition_path, "090")

                # Load silhouette sequence
                silhouette_sequence = load_silhouette_sequence(sequence_path)

                if silhouette_sequence:
                    # Create GEI
                    gei = processor.process_silhouette_sequence(silhouette_sequence)

                    if gei is not None:
                        # Determine if this condition goes to train or test
                        is_train = condition in train_conditions
                        output_subdir = "train" if is_train else "test"

                        # Save GEI
                        output_filename = f"{subject_id}_{condition}.png"
                        output_path = os.path.join(
                            output_dir, output_subdir, output_filename
                        )
                        cv2.imwrite(output_path, gei)

                        # Update counters
                        if is_train:
                            total_train_samples += 1
                            subject_stats[subject_id]["train"] += 1
                        else:
                            total_test_samples += 1
                            subject_stats[subject_id]["test"] += 1

    # Print summary statistics
    print(f"\nDataset Summary:")
    print(f"Total training samples: {total_train_samples}")
    print(f"Total testing samples: {total_test_samples}")

    # Check subject distribution
    subjects_with_both = sum(
        1 for s in subject_stats.values() if s["train"] > 0 and s["test"] > 0
    )
    print(
        f"Subjects with both training and testing samples: {subjects_with_both}/{len(subject_stats)}"
    )

    # Save subject distribution to a log file
    with open(os.path.join(output_dir, "dataset_stats.txt"), "w") as f:
        f.write(f"Subject ID,Training Samples,Testing Samples\n")
        for subject_id, counts in subject_stats.items():
            f.write(f"{subject_id},{counts['train']},{counts['test']}\n")

    print(
        f"Dataset statistics saved to {os.path.join(output_dir, 'dataset_stats.txt')}"
    )


if __name__ == "__main__":
    # Update these paths to match your dataset location
    ROOT_DATASET_DIR = "C:\\Users\\fwiko\\Downloads\\GaitDatasetB-silh"
    OUTPUT_DIR = "output"

    # Set a seed for reproducibility
    random.seed(42)

    prepare_dataset(ROOT_DATASET_DIR, OUTPUT_DIR, train_split=0.5)
