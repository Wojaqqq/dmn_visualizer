import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob


def find_arrow_direction(self, img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Apply binary thresholding
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming that the largest contour is the arrow
    cnt = max(contours, key=cv2.contourArea)

    # Flatten the contour array and use PCA to find the main direction
    data = cnt.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(data, mean=None)

    # The principal eigenvector gives the direction
    principal_eigenvector = eigenvectors[0]

    return principal_eigenvector


def find_nearest_elements(self, point, direction, elements):
    nearest_element_start = None
    nearest_element_end = None
    smallest_distance_start = np.inf
    smallest_distance_end = np.inf

    for element in elements:
        # Calculate the center of the element's bbox
        center = ((element.bbox[0] + element.bbox[2]) / 2,
                  (element.bbox[1] + element.bbox[3]) / 2)

        # Calculate the vector from the point to the element
        vector = (center[0] - point[0], center[1] - point[1])

        # Calculate the angle to the element
        angle = np.arccos(np.dot(direction, vector) /
                          (np.linalg.norm(direction) * np.linalg.norm(vector)))

        # If the angle is too large, ignore the element
        if np.abs(angle) > np.pi / 2:
            continue

        # Calculate the distance to the element
        distance = np.linalg.norm(vector)

        # Update the nearest element if this element is closer in the direction
        # Assign the element to the start if it's on the left, and to the end if it's on the right
        if vector[0] < 0 and distance < smallest_distance_start:
            smallest_distance_start = distance
            nearest_element_start = element
        elif vector[0] > 0 and distance < smallest_distance_end:
            smallest_distance_end = distance
            nearest_element_end = element

    return nearest_element_start, nearest_element_end


def bound_arrow_elements(self):
    # List of elements to bind arrows to
    elements_to_bind = self.inputs + self.decisions + self.knowledge_model + self.knowledge_source

    for arrow in self.arrows:
        image = cv2.imread(str(self.img_path), cv2.IMREAD_GRAYSCALE)
        bbox = list(arrow.bbox)
        crop_img = image[bbox[1] - self.crop_threshold: bbox[3] + self.crop_threshold,
                   bbox[0] - self.crop_threshold: bbox[2] + self.crop_threshold]

        # Find the direction of the arrow
        direction = self.find_arrow_direction(crop_img)

        # Use the midpoint of the arrow's bounding box as the starting point
        mid_point = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        # Find the nearest elements to the start and end of the arrow
        nearest_to_start, nearest_to_end = self.find_nearest_elements(mid_point, direction, elements_to_bind)

        # Bound the arrow to the nearest elements
        arrow.source = nearest_to_start
        arrow.target = nearest_to_end



# master_path =  r'C:\Users\ttwoj\PycharmProjects\_main\dmn_visualizer\logs\arrows'
# for elem in glob.glob(master_path + '/*.jpg'):
elem = r'C:\Users\ttwoj\PycharmProjects\_main\dmn_visualizer\logs\arrows\arrow_1690405096.89266.jpg'
print(elem)
plt.figure(figsize=(10, 10))
arrow_path = r'C:\Users\ttwoj\PycharmProjects\_main\dmn_visualizer\logs\arrows\arrow_1690405096.89266.jpg'
# , cv2.IMREAD_GRAYSCALE
image = cv2.imread(elem)
print(find_arrow_direction(image))

# xx, xy = x
# yy, yx = y
# image = cv2.circle(image, (xx, xy), radius=0, color=(255, 0, 0), thickness=5)  # tail - red
# image = cv2.circle(image, (yy, yx), radius=0, color=(0, 255, 0), thickness=5)  # head - green
# plt.imshow(image)
# plt.show()
