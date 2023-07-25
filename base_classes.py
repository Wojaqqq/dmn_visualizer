import os
import xml.etree.ElementTree as ET
from get_bboxes import detect_bboxes
from utils import load_label_map, simpleHTR_word_detector, make_dir
import cv2
import time


class Element:
    def __init__(self, _id, name, bbox, img, connection=None):
        self._id = _id
        self.bbox = bbox
        self.img = img
        self.crop_threshold = 20
        self.save_crop_img = True
        self.connection=connection
        self.name = self.extract_text(name)

    @property
    def id(self):
        return self._id

    def extract_text(self, name):
        if name.startswith('text'):
            image = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)
            crop_img = image[self.bbox[1] - self.crop_threshold: self.bbox[3] + self.crop_threshold,
                             self.bbox[0] - self.crop_threshold: self.bbox[2] + self.crop_threshold]

            make_dir('tmp')
            crop_image_path = f"./tmp/text_crop_image_{time.time()}.jpg"
            cv2.imwrite(crop_image_path, crop_img)
            text, probability = simpleHTR_word_detector(crop_image_path)
            if not self.save_crop_img:
                os.remove(crop_image_path)
            if float(probability) > 0.1:
                return text
            else:
                return name
        else:
            return name

    def __str__(self):
        return f"Element(Id: {self._id}, Name: {self.name}, Bbox: {self.bbox})"


class GraphImage:
    def __init__(self, img, model, label_map, confidence_threshold=0.3, verbose=False, input_width=640,
                 input_height=640):
        self.img = img
        self.model = model
        self.label_map = label_map
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.input_width = input_width
        self.input_height = input_height
        self.detected_data = detect_bboxes(self.img, self.model, self.label_map, self.confidence_threshold,
                                           self.input_width, self.input_height, self.verbose)

    def parse_elements(self):
        elements = []
        lbl_map = load_label_map(self.label_map)
        class_counter = {elem: 0 for elem in lbl_map.values()}
        for data in self.detected_data:
            tag, *bbox = data
            tag = f"{tag}_{class_counter[tag]}"
            elements.append(Element(tag, tag, bbox, self.img))
            class_counter[data[0]] += 1
        return elements


class XMLParser:
    def __init__(self, file_path, img):
        self.file_path = file_path
        self.root = self.load_xml()
        self.img = img

    def load_xml(self):
        tree = ET.parse(self.file_path)
        return tree.getroot()

    def parse_elements(self):
        elements = []
        object_dict = {}

        for obj in self.root.findall('.//object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            bounding_box = (xmin, ymin, xmax, ymax)
            if name in object_dict:
                object_dict[name].append(bounding_box)
            else:
                object_dict[name] = [bounding_box]

        for key, values in object_dict.items():
            for i, bbox in enumerate(values):
                tag = f"{key}_{i}"
                elements.append(Element(tag, tag, bbox, self.img))

        return elements
