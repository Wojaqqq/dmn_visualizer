import os
import xml.etree.ElementTree as ET
from get_bboxes import detect_bboxes
from utils import load_label_map, simpleHTR_word_detector, make_dir
from spellchecker import SpellChecker
import cv2
import time
from pathlib import Path
import logging

logger = logging.getLogger()


class Element:
    def __init__(self, _id, name, bbox, img, source=None, target=None):
        self._id = _id
        self.bbox = bbox
        self.img_path = img
        self.crop_threshold = 10
        self.word_probability_threshold = 0.1
        self.save_crop_img = False
        self.source = source
        self.target = target
        self.name = self.extract_text(name)

    @property
    def id(self):
        return self._id

    def extract_text(self, name):
        if name.startswith('text'):
            image = cv2.imread(str(self.img_path), cv2.IMREAD_GRAYSCALE)
            bbox = list(self.bbox)
            crop_img = image[bbox[1] - self.crop_threshold: bbox[3] + self.crop_threshold,
                             bbox[0] - self.crop_threshold: bbox[2] + self.crop_threshold]

            make_dir('tmp')
            crop_img_path = f"./tmp/text_crop_image_{time.time()}_{Path(self.img_path).stem}.jpg"
            cv2.imwrite(crop_img_path, crop_img)
            text, probability = simpleHTR_word_detector(crop_img_path)
            logger.info(f"Detected text: {text} in element: {self.id} with probability: {probability}")
            if not self.save_crop_img:
                os.remove(crop_img_path)
            if float(probability) > self.word_probability_threshold:
                spell = SpellChecker()
                corrected_sentence = " ".join([spell.correction(w) for w in text.split() if spell.correction(w) is not None])
                logger.info(f"Text: {text} changed with spellchecker to {corrected_sentence}")
                return corrected_sentence
            else:
                logger.info(f"Detected text: {text} with probability: {probability} - cannot replace")
                return name
        return name

    def __str__(self):
        if self.source is None and self.target is None:
            return f"Element(Id: {self._id}, Name: {self.name}, Bbox: {self.bbox})"
        return f"Element(Id: {self._id}, Name: {self.name}, Bbox: {self.bbox}, Source: {self.source}, Target: {self.target})"


class TestElement(Element):
    def __init__(self, _id, name, bbox, img, source=None, target=None):
        super().__init__(_id, name, bbox, img, source, target)
        self.name = name


class GraphImage:
    def __init__(self, img_path, model, label_map, confidence_threshold=0.3, verbose=False, input_width=640,
                 input_height=640):
        self.img_path = img_path
        self.model = model
        self.label_map = label_map
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.input_width = input_width
        self.input_height = input_height
        self.detected_data = detect_bboxes(self.img_path, self.model, self.label_map, self.confidence_threshold,
                                           self.input_width, self.input_height, self.verbose)

    def parse_elements(self):
        elements = []
        lbl_map = load_label_map(self.label_map)
        class_counter = {elem: 0 for elem in lbl_map.values()}
        for data in self.detected_data:
            tag, bbox = data
            tag = f"{tag}_{class_counter[tag]}"
            elements.append(Element(tag, tag, bbox, self.img_path))
            class_counter[data[0]] += 1
        return elements


class XMLParser:
    def __init__(self, file_path, img):
        self.file_path = file_path
        self.root = self.load_xml()
        self.img_path = img

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
                elements.append(Element(tag, tag, bbox, self.img_path))

        return elements
