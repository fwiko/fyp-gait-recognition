from collections import deque

import cv2
import numpy as np


class GaitProcessor:
    NORMALISED_WIDTH = 88
    NORMALISED_HEIGHT = 128
    MIN_CONTOUR_AREA = 500
    MIN_ASPECT_RATIO = 1.5
    MAX_ASPECT_RATIO = 2.5

    def __init__(self, initial_frames=30, buffer_size=30, alpha=0.2):
        self.background_model = None
        self.initial_frames_count = initial_frames
        self.background_frames = []
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.aligned_buffer = deque(maxlen=buffer_size) if buffer_size > 0 else []
        self.aligned_gei = None

    def initialise_backround(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.background_frames.append(gray)

        if len(self.background_frames) >= self.initial_frames_count:
            self.background_model = np.median(self.background_frames, axis=0).astype(
                np.uint8
            )
            self.background_frames = []
            return True

        return False

    def reset_background(self):
        self.background_model = None
        self.background_frames.clear()
        self.aligned_gei = None

        if hasattr(self.aligned_buffer, "clear"):
            self.aligned_buffer.clear()
        else:
            self.aligned_buffer = []

    def process_frame(self, frame):
        if self.background_model is None:
            return None, None

        silhouette, bounding_box = self._extract_silhouette(frame)

        if bounding_box is not None:
            self._draw_bounding_box(frame, bounding_box)

            aligned_silhouette = self._align_silhouette(silhouette, bounding_box)
            if aligned_silhouette is not None:
                self._update_gei(aligned_silhouette)

        if self.aligned_gei is not None:
            gei = (self.aligned_gei * 255).astype(np.uint8)
        else:
            gei = np.zeros(
                (self.NORMALISED_HEIGHT, self.NORMALISED_WIDTH), dtype=np.uint8
            )

        return silhouette, gei

    def _extract_silhouette(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        diff = cv2.absdiff(self.background_model, gray)
        _, binary = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        silhouette = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )

        contours, _ = cv2.findContours(
            silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        large_contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) > self.MIN_CONTOUR_AREA
        ]

        filtered_silhouette = np.zeros_like(silhouette)
        cv2.drawContours(filtered_silhouette, large_contours, -1, 255, -1)

        bounding_box = None
        if large_contours:
            largest_contour = max(large_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            aspect_ratio = h / w
            if self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO:
                bounding_box = (x, y, w, h)

        return filtered_silhouette, bounding_box

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

    def _update_gei(self, aligned_silhouette):
        if isinstance(self.aligned_buffer, deque):
            self.aligned_buffer.append(aligned_silhouette)
        else:
            self.aligned_buffer.append(aligned_silhouette)

        if self.buffer_size == 0 or len(self.aligned_buffer) == self.buffer_size:
            if self.aligned_gei is None:
                self.aligned_gei = np.mean(self.aligned_buffer, axis=0)
            else:
                new_gei = np.mean(self.aligned_buffer, axis=0)
                self.aligned_gei = (
                    1 - self.alpha
                ) * self.aligned_gei + self.alpha * new_gei

    def _draw_bounding_box(self, frame, bounding_box):
        x, y, w, h = bounding_box
        aspect_ratio = h / w
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label = f"AR: {aspect_ratio:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x, text_y = (x + 5, y + text_size[1] + 5)

        cv2.putText(
            frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness
        )
