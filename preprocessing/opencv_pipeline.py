import cv2
import numpy as np

class MRIProcessor:
    def __init__(self, img_size=224):
        self.img_size = img_size

    def load_from_bytes(self, image_bytes: bytes):
        """Load MRI image from raw web bytes instead of a disk path"""
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image not found or invalid format")
        return img

    def convert_grayscale(self, img):
        """Convert to grayscale (MRI standard preprocessing)"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def resize_image(self, img):
        """Resize image to model input size"""
        return cv2.resize(img, (self.img_size, self.img_size))

    def denoise(self, img):
        """Remove noise using Gaussian Blur"""
        return cv2.GaussianBlur(img, (5, 5), 0)

    def normalize(self, img):
        """Normalize pixel values to 0–1"""
        return img / 255.0

    def preprocess(self, image_bytes: bytes):
        """Full preprocessing pipeline"""
        img = self.load_from_bytes(image_bytes)
        img = self.convert_grayscale(img)
        img = self.resize_image(img)
        img = self.denoise(img)
        img = self.normalize(img)

        # reshape for CNN input: (224,224,1)
        img = np.expand_dims(img, axis=-1)

        return img