import cv2
import numpy as np
import os

# ----------------------- CONFIGURATION -----------------------
# Paths to YOLO files
config_path = "yolov4.cfg"           # Path to yolov4.cfg
weights_path = "yolov4.weights"      # Path to yolov4.weights
names_path = "classes.names"         # Custom class file (e.g., containing "tooth")

# Image path
image_path = "data/images/bitewing.jpg"  # Change this to your input image path

# Confidence and threshold
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3
# -------------------------------------------------------------

# Load class labels
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLOv4 model
net = cv2.dnn.readNet(weights_path, config_path)

# Enable CUDA if available
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load image
image = cv2.imread(image_path)
height, width = image.shape[:2]

# Create input blob
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Run forward pass
layer_outputs = net.forward(output_layers)

# Collect detection data
boxes = []
confidences = []
class_ids = []

for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONFIDENCE_THRESHOLD:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

# Draw boxes
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    color = (0, 255, 0)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show image
cv2.imshow("Teeth Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
