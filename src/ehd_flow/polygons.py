"""PolygonExtractor: turn blobs in an image into polygon obstacles.

Single responsibility: image -> polygons. This is the one implementation
used by the ``image_to_polygons`` CLI wrapper *and* imported by
:mod:`ehd_flow.coulomb` -- two front ends, one helper class.

Pipeline:
  1. Load the input image and convert to grayscale.
  2. Binarize: pixel values < 255/2 become 0, values > 255/2 become 1
     (a pixel exactly at 127.5 maps to 0).
  3. Find the external contour of every connected blob of 1-valued pixels.
     Holes are ignored, so a doughnut-shaped blob becomes a plain circle.
  4. Simplify each contour into a polygon (Douglas-Peucker).
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def show_image(path, title):
    """Open an image in a preview window; no-op message when headless."""
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        cv2.imshow(title, img)
        print("close the preview window to finish (or press any key)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print(f"no display available; image saved to {path} instead")


class PolygonExtractor:
    """Extracts simplified polygons from blobs in an image.

    Parameters mirror the old image_to_polygons.py flags:
      epsilon_ratio -- Douglas-Peucker simplification as a fraction of the
                       contour perimeter.
      min_area      -- ignore blobs smaller than this many px^2.
    """

    THRESHOLD = 255 / 2  # 127.5

    def __init__(self, epsilon_ratio=0.002, min_area=10.0):
        if cv2 is None:
            raise ImportError(
                "opencv-python is required (pip install opencv-python)")
        self.epsilon_ratio = epsilon_ratio
        self.min_area = min_area

    # -- image loading / binarization -----------------------------------

    def read_grayscale(self, path):
        """Load an image path as a single-channel uint8 grayscale array."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        if img.ndim == 3:
            img = img[:, :, :3]  # drop alpha if present
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def binarize(self, gray):
        """Binarize a grayscale uint8 array: > 255/2 -> 1, else 0."""
        _, binary = cv2.threshold(gray, self.THRESHOLD, 1,
                                  cv2.THRESH_BINARY)
        return binary.astype(np.uint8)

    def load_binary(self, path):
        """Load image, grayscale it, binarize at 255/2 -> {0, 1} uint8."""
        return self.binarize(self.read_grayscale(path))

    # -- polygon extraction --------------------------------------------

    def extract_polygons(self, binary, scale_x=1.0, scale_y=1.0):
        """Return a list of (name, points) polygons from the binary image."""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # deterministic order: biggest blob first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        polygons = []
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < self.min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.epsilon_ratio * peri, True)
            pts = approx.reshape(-1, 2)
            if len(pts) < 3:
                continue
            pts = pts.astype(float)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            polygons.append((f"blob_{i}", pts.tolist()))
        return polygons

    # -- preview rendering ----------------------------------------------

    def draw_preview(self, binary, polygons, scale_x=1.0, scale_y=1.0):
        """Black-and-white image with each polygon drawn as a border."""
        canvas = (binary * 255).astype(np.uint8)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        for _, pts in polygons:
            arr = np.asarray(pts, dtype=float)
            arr[:, 0] /= scale_x
            arr[:, 1] /= scale_y
            cv2.polylines(canvas, [arr.astype(np.int32)], True,
                          (0, 255, 0), 2)
        return canvas

    def save_preview(self, path, preview):
        """Write a preview canvas to an image file."""
        cv2.imwrite(path, preview)
