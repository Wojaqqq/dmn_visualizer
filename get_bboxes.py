import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from pathlib import Path
from utils import load_label_map, make_dir


def process_image(image, width, height):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    if image.shape[:2] != (height, width):
        resized_image = cv2.resize(image, (width, height))
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    return input_tensor, resized_image


def detect_bboxes(img_path, model, label_map, confidence_threshold, input_width, input_height, verbose):
    model = tf.saved_model.load(model)
    label_map = load_label_map(label_map)
    image = cv2.imread(str(img_path))
    input_tensor, resized_image = process_image(image, input_width, input_height)
    detections = model(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    scores = detections['detection_scores']

    detected_data = []
    image = np.array(image)
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            box = boxes[i]
            class_id = classes[i]
            score = scores[i]
            y_min, x_min, y_max, x_max = box
            y_min *= image.shape[0]
            y_max *= image.shape[0]
            x_min *= image.shape[1]
            x_max *= image.shape[1]

            bbox_info = (label_map[class_id], (int(x_min), int(y_min), int(x_max), int(y_max)))
            detected_data.append(bbox_info)
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 1)
            cv2.putText(image, f"{label_map[class_id]}, {int(x_min), int(y_min), int(x_max), int(y_max)}", (int(x_min), int(y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)
            if verbose:
                print(f'Class: {label_map[class_id]}, Score: {score}, BBox: ({int(x_min), int(y_min), int(x_max), int(y_max)})')
    plt.figure(figsize=(10, 10))
    make_dir('output')
    # TODO name with time.time - img_path only for testing purposes
    # cv2.imwrite(fr'output/detections_{time.time()}.jpg', image)
    cv2.imwrite(fr'output/detections_{Path(img_path).stem}.jpg', image)

    if verbose:
        plt.imshow(image)
        plt.show()
    return detected_data


if __name__ == "__main__":
    """
    Sample parameters for test run
    """
    image_path = r'data/dmn_ex01_001.jpeg'
    model = r'model/saved_model'
    label_map = r'model/label_map.pbtxt'
    confidence_threshold = 0.5
    input_width = 640
    input_height = 640
    verbose = True
    detect_bboxes(image_path, model, label_map, confidence_threshold, input_width, input_height, verbose)
