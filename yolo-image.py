from ultralytics import YOLO
import cv2

model = YOLO('../yolo-weighs/yolov8l.pt')

# Disable internal display
results = model("chap-5-yolo/school.webp", show=False)
#When you run inference using Ultralytics YOLO, the model() call returns a list of Results
#results accesses the first (or only) image's result.

#generates a NumPy array with bounding boxes, class labels, and confidence scores drawn on the input image.
annotated_img = results[0].plot()

# Show with OpenCV window which you control
cv2.imshow("Detection", annotated_img)
cv2.waitKey(0)  # Wait for key press indefinitely
cv2.destroyAllWindows()  # Close window after key press
