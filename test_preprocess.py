from preprocessing.opencv_pipeline import MRIProcessor
import cv2

processor = MRIProcessor()

# change this path to your local MRI image
img_path = "sample_mri.jpg"

try:
    output = processor.preprocess(img_path)
    print("Preprocessing successful!")
    print("Shape:", output.shape)
    print("Max pixel:", output.max())
    print("Min pixel:", output.min())

except Exception as e:
    print("Error:", e)