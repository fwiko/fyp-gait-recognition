import cv2
import numpy as np
from collections import deque

NORMALIZED_SIZE = (88, 128)
MAX_GEI_FRAMES = 30

gei_frames = deque(maxlen=MAX_GEI_FRAMES)


class GaitProcessor:
    def __init__(self, initial_frames=30, buffer_size=30, alpha=0.2):
        self.static_background = None
        self.initial_frames = initial_frames
        self.background_buffer = []
        self.buffer_size = buffer_size
        self.alpha = alpha

        if buffer_size == 0:
            self.silhouette_buffer = []
        else:
            self.silhouette_buffer = deque(maxlen=buffer_size)

        self.gait_energy_image = None

    def initialize_background(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.background_buffer.append(gray)

        if len(self.background_buffer) >= self.initial_frames:
            self.static_background = np.median(self.background_buffer, axis=0).astype(
                np.uint8
            )
            self.background_buffer = []
            return True

        return False

    def reset_background(self):
        self.gait_energy_image = None
        self.background_buffer.clear()

    def process_frame(self, frame):
        if self.static_background is None:
            return None, None, None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        diff = cv2.absdiff(self.static_background, gray)

        _, binary = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        silhouette = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )

        contours, _ = cv2.findContours(
            silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

        clean_silhouette = np.zeros_like(silhouette)
        cv2.drawContours(clean_silhouette, large_contours, -1, 255, -1)

        color_silhouette = np.zeros_like(frame)
        color_silhouette[clean_silhouette == 255] = (255, 255, 255)

        if large_contours:
            largest_contour = max(large_contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(largest_contour)

            bounding_colour = (0, 255, 0) if h > w else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), bounding_colour, 2)
            cv2.rectangle(color_silhouette, (x, y), (x + w, y + h), bounding_colour, 2)

            mask = np.zeros(color_silhouette.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -2)

            color_silhouette = cv2.bitwise_and(
                color_silhouette, color_silhouette, mask=mask
            )

            if h > w:
                person_roi = clean_silhouette[y : y + h, x : x + w]

                ty, tx = np.where(person_roi > 0)

                if len(ty) > 0 and len(tx) > 0:
                    sy, ey = ty.min(), ty.max() + 1
                    sx, ex = tx.min(), tx.max() + 1

                    silhouette_h = ey - sy
                    silhouette_w = ex - sx

                    if silhouette_h > silhouette_w:
                        cx = int(tx.mean())

                        cenX = silhouette_h // 2
                        start_w = (silhouette_h - silhouette_w) // 2

                        if max(cx - sx, ex - cx) < cenX:
                            start_w = cenX - (cx - sx)

                        square_silhouette = np.zeros(
                            (silhouette_h, silhouette_h), np.uint8
                        )
                        square_silhouette[:, start_w : start_w + silhouette_w] = (
                            person_roi[sy:ey, sx:ex]
                        )

                        offsetX = 20
                        resized_silhouette = cv2.resize(
                            square_silhouette, (128, 128), interpolation=cv2.INTER_AREA
                        )
                        normalized_silhouette = resized_silhouette[
                            :, offsetX : offsetX + 88
                        ]

                        if not hasattr(self, "normalized_buffer"):
                            if self.buffer_size == 0:
                                self.normalized_buffer = []
                            else:
                                self.normalized_buffer = deque(maxlen=self.buffer_size)

                        normalized_frame = normalized_silhouette.astype(float) / 255.0
                        if self.buffer_size == 0:
                            self.normalized_buffer.append(normalized_frame)
                        else:
                            self.normalized_buffer.append(normalized_frame)

                        if not hasattr(self, "normalized_gei"):
                            self.normalized_gei = None

                        if (
                            self.buffer_size == 0
                            or len(self.normalized_buffer) == self.buffer_size
                        ):
                            if self.normalized_gei is None:
                                self.normalized_gei = np.mean(
                                    self.normalized_buffer, axis=0
                                )
                            else:
                                new_norm_gei = np.mean(self.normalized_buffer, axis=0)
                                self.normalized_gei = (
                                    1 - self.alpha
                                ) * self.normalized_gei + self.alpha * new_norm_gei

        if self.buffer_size == 0 or len(self.silhouette_buffer) == self.buffer_size:
            if self.gait_energy_image is None:
                self.gait_energy_image = np.mean(self.silhouette_buffer, axis=0)
            else:
                new_gei = np.mean(self.silhouette_buffer, axis=0)
                self.gait_energy_image = (
                    1 - self.alpha
                ) * self.gait_energy_image + self.alpha * new_gei

        norm_gei = None
        if hasattr(self, "normalized_gei") and self.normalized_gei is not None:
            norm_gei = (self.normalized_gei * 255).astype(np.uint8)
        else:
            norm_gei = np.zeros((128, 64, 3), dtype=np.uint8)

        return color_silhouette, norm_gei
