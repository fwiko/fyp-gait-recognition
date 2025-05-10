from collections import deque

import cv2
import numpy as np


class GaitProcessor:
    NORMALISED_WIDTH = 88
    NORMALISED_HEIGHT = 128
    MIN_CONTOUR_AREA = 500
    MIN_ASPECT_RATIO = 1.5
    MAX_ASPECT_RATIO = 3.5

    def __init__(self, initial_frames=30, buffer_size=30, alpha=0.2):
        self.background_model = None
        self.initial_frames_count = initial_frames
        self.background_frames = []
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.aligned_buffer = deque(maxlen=buffer_size) if buffer_size > 0 else []
        self.aligned_gei = None

    def initialise_backround(self, frame):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Append the grayscale frame to the background frames
        self.background_frames.append(gray)

        # If the number of background frames is greater than or equal to the initial frames count (30), initialise the background model
        if len(self.background_frames) >= self.initial_frames_count:
            # Calculate the median of the background frames
            self.background_model = np.median(self.background_frames, axis=0).astype(
                np.uint8
            )

            # Clear the background frames
            self.background_frames = []
            return True

        return False

    def reset_background(self):
        self.background_model = None
        self.background_frames.clear()
        self.aligned_gei = None
        self.aligned_buffer.clear()

    def process_frame(self, frame):
        if self.background_model is None:
            return None, None

        # Extract the silhouette and bounding box from the frame    
        silhouette, bounding_box = self._extract_silhouette(frame)

        # If a bounding box is found, draw it on the frame and align the silhouette
        if bounding_box is not None:
            self._draw_bounding_box(frame, bounding_box)

            aligned_silhouette = self._align_silhouette(silhouette, bounding_box)
            if aligned_silhouette is not None:
                self._update_gei(aligned_silhouette)

        if self.aligned_gei is not None:
            gei = (self.aligned_gei * 255).astype(np.uint8)
        else:
            # If no bounding box is found, set the GEI to a zero array
            gei = np.zeros(
                (self.NORMALISED_HEIGHT, self.NORMALISED_WIDTH), dtype=np.uint8
            )

        return silhouette, gei

    def _extract_silhouette(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0) # Blur the grayscale frame

        diff = cv2.absdiff(self.background_model, gray) # Calculate the absolute difference between the background model and the grayscale frame
        _, binary = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY) # Threshold the difference (i.e. where the difference is greater than 25, set to 255, otherwise 0)

        # Apply morphological closing to the binary image to fill in holes
        silhouette = cv2.morphologyEx( 
            binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )

        # Find the contours in the binary image
        contours, _ = cv2.findContours(
            silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter the contours to only include the largest ones
        large_contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) > self.MIN_CONTOUR_AREA
        ]

        # Draw the largest contours on a new frame array
        filtered_silhouette = np.zeros_like(silhouette)
        cv2.drawContours(filtered_silhouette, large_contours, -1, 255, -1)

        bounding_box = None
        
        # If there are large contours, find the largest one and calculate its bounding box
        if large_contours:
            largest_contour = max(large_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            aspect_ratio = h / w
            if self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO:
                bounding_box = (x, y, w, h)

        return filtered_silhouette, bounding_box

    def _align_silhouette(self, silhouette, bounding_box):
        x, y, w, h = bounding_box # Extract the bounding box coordinates

        person_region = silhouette[y : y + h, x : x + w] # Extract the person region (region of interest)from the silhouette

        ty, tx = np.where(person_region > 0) # Find the coordinates of the non-zero pixels

        if len(ty) == 0 or len(tx) == 0:
            return None

        sy, ey = ty.min(), ty.max() + 1 # Calculate the start and end y-coordinates of the silhouette
        sx, ex = tx.min(), tx.max() + 1 # Calculate the start and end x-coordinates of the silhouette

        silhouette_h = ey - sy # Calculate the height of the silhouette
        silhouette_w = ex - sx # Calculate the width of the silhouette

        if silhouette_h <= silhouette_w: 
            return None

        cx = int(tx.mean()) # The x-coordinate of the centre of non-zero pixels in the silhouette
        cenX = silhouette_h // 2 # The centre of the silhouette
        start_w = (silhouette_h - silhouette_w) // 2

        # If the distance between the centre of the silhouette and the centre of the non-zero pixels is less than the centre of the silhouette, set the starting width to the distance between the centre of the silhouette and the centre of the non-zero pixels
        if max(cx - sx, ex - cx) < cenX: 
            start_w = cenX - (cx - sx)

        square_silhouette = np.zeros((silhouette_h, silhouette_h), np.uint8) # Create a new frame array of the same size as the silhouette
        square_silhouette[:, start_w : start_w + silhouette_w] = person_region[
            sy:ey, sx:ex
        ] # Extract the person region from the silhouette and place it in the centre of the new frame array

        resized_silhouette = cv2.resize(
            square_silhouette,
            (self.NORMALISED_HEIGHT, self.NORMALISED_HEIGHT),
            interpolation=cv2.INTER_AREA,
        ) # Resize the silhouette to the normalised size

        offsetX = 20 # The offset of the silhouette from the left edge of the frame
        normalised_silhouette = resized_silhouette[
            :, offsetX : offsetX + self.NORMALISED_WIDTH
        ] # Crop the silhouette to the normalised width and height while maintaining alignment

        return normalised_silhouette.astype(float) / 255.0 # Normalise the silhouette to the range 0-1

    def _update_gei(self, aligned_silhouette):
        self.aligned_buffer.append(aligned_silhouette) # Append the aligned silhouette to the buffer

        # If the buffer size is 0 or the buffer is full, calculate the mean of the aligned silhouettes (GEI)
        if self.buffer_size == 0 or len(self.aligned_buffer) == self.buffer_size: 
            if self.aligned_gei is None: 
                self.aligned_gei = np.mean(self.aligned_buffer, axis=0) # If the GEI is not set, set it to the mean of the aligned silhouettes
            else: # Otherwise, update the GEI using a weighted (exponential moving) average, prioritising older frames (self.alpha = 0.2)
                new_gei = np.mean(self.aligned_buffer, axis=0)
                self.aligned_gei = (
                    1 - self.alpha
                ) * self.aligned_gei + self.alpha * new_gei

    def _draw_bounding_box(self, frame, bounding_box):
        x, y, w, h = bounding_box # Extract the bounding box coordinates

        aspect_ratio = h / w # Calculate the aspect ratio of the bounding box

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Draw the bounding box on the frame

        # The label to display on the frame denoting the aspect ratio of the bounding box
        label = f"AR: {aspect_ratio:.2f}" 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x, text_y = (x + 5, y + text_size[1] + 5)

        cv2.putText(
            frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness
        )
