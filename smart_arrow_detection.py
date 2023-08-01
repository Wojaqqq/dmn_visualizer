import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import time


def find_head_and_tail(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Apply binary thresholding
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming that the largest contour is the arrow
    cnt = max(contours, key=cv2.contourArea)

    # Calculate the centroid of the contour
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroid = (cX, cY)

    # Calculate the distances of all contour points to the centroid
    distances = np.sqrt((cnt[:, :, 0] - cX) ** 2 + (cnt[:, :, 1] - cY) ** 2)

    # The tail is the point nearest to the centroid
    tail = tuple(cnt[distances.argmin()][0])

    # The head is the point farthest from the centroid
    head = tuple(cnt[distances.argmax()][0])

    return tail, head


def is_within_bbox(self, point, bbox):
    x, y = point
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def find_nearest_element(self, point, direction, elements):
    nearest_element = None
    smallest_angle = np.inf

    for element in elements:
        # Calculate the center of the element's bbox
        center = ((element.bbox[0] + element.bbox[2]) / 2,
                  (element.bbox[1] + element.bbox[3]) / 2)

        # Calculate the vector from the point to the element
        vector = (center[0] - point[0], center[1] - point[1])

        # Calculate the angle to the element
        angle = np.arccos(np.dot(direction, vector) /
                          (np.linalg.norm(direction) * np.linalg.norm(vector)))

        # Update the nearest element if this element is closer in the direction
        if angle < smallest_angle:
            smallest_angle = angle
            nearest_element = element

    return nearest_element


def bound_arrow_elements(self):
    # List of elements to bind arrows to
    elements_to_bind = self.inputs + self.decisions + self.knowledge_model + self.knowledge_source

    for arrow in self.arrows:
        image = cv2.imread(str(self.img_path), cv2.IMREAD_GRAYSCALE)
        bbox = list(arrow.bbox)
        crop_img = image[bbox[1] - self.crop_threshold: bbox[3] + self.crop_threshold,
                   bbox[0] - self.crop_threshold: bbox[2] + self.crop_threshold]
        cv2.imwrite(rf'logs/arrows/to_detect_arrow_{time.time()}', crop_img)

        # Find the head and tail of the arrow
        head, tail = self.find_head_and_tail(crop_img)

        # Calculate the coordinates of the head and tail relative to the original image
        head = (head[0] + bbox[0] - self.crop_threshold, head[1] + bbox[1] - self.crop_threshold)
        tail = (tail[0] + bbox[0] - self.crop_threshold, tail[1] + bbox[1] - self.crop_threshold)

        # Calculate the direction from the head to the tail
        direction_head_to_tail = (tail[0] - head[0], tail[1] - head[1])
        direction_tail_to_head = (head[0] - tail[0], head[1] - tail[1])

        # Find the nearest elements to the head and tail in their respective directions
        nearest_to_head = self.find_nearest_element(head, direction_head_to_tail, elements_to_bind)
        nearest_to_tail = self.find_nearest_element(tail, direction_tail_to_head, elements_to_bind)

        # Bound the arrow to the nearest elements
        arrow.source = nearest_to_tail
        arrow.target = nearest_to_head


######## for testing purposes ########
if __name__ == "__main__":
    master_path = r'C:\Users\ttwoj\PycharmProjects\_main\dmn_visualizer\logs\arrows'
    for elem in glob.glob(master_path + '/*.jpg'):
        print(elem)
        plt.figure(figsize=(10, 10))
        arrow_path = r'C:\Users\ttwoj\PycharmProjects\_main\dmn_visualizer\logs\arrows\arrow_1690405096.89266.jpg'
        # , cv2.IMREAD_GRAYSCALE
        image = cv2.imread(elem)
        x, y = find_head_and_tail(image)
        xx, xy = x
        yy, yx = y
        image = cv2.circle(image, (xx, xy), radius=0, color=(255, 0, 0), thickness=5)  # tail - red
        image = cv2.circle(image, (yy, yx), radius=0, color=(0, 255, 0), thickness=5)  # head - green
        plt.imshow(image)
        plt.show()
