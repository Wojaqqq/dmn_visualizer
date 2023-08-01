import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from utils import make_dir
from collections import defaultdict
from config import ElementsType
import numpy as np
import cv2
import time


class XMLCreator:
    def __init__(self, elements, name, output_path, img_path):
        self.name = name
        self.img_path = img_path
        self.texts, self.arrows, self.inputs, self.decisions, self.knowledge_model, self.knowledge_source = self.split_elements(
            elements)
        self.output_path = self.create_output(output_path)
        self.crop_threshold = 10
        self.objects = {e.id: e for e in self.arrows + self.inputs + self.decisions + self.knowledge_model + self.knowledge_source}

        self.xmlns = {"": "https://www.omg.org/spec/DMN/20191111/MODEL/",
                      "dmndi": "https://www.omg.org/spec/DMN/20191111/DMNDI/",
                      "dc": "http://www.omg.org/spec/DMN/20191111/DC/",
                      "di": "http://www.omg.org/spec/DMN/20191111/DI/"}

        ET.register_namespace("", self.xmlns[""])
        ET.register_namespace("dmndi", self.xmlns["dmndi"])
        ET.register_namespace("dc", self.xmlns["dc"])
        ET.register_namespace("di", self.xmlns["di"])
        # TODO delete arrow image saving
        # for arrow in self.arrows:
        #     image = cv2.imread(str(self.img_path), cv2.IMREAD_GRAYSCALE)
        #     bbox = list(arrow.bbox)
        #     crop_img = image[bbox[1] - self.crop_threshold: bbox[3] + self.crop_threshold,
        #                bbox[0] - self.crop_threshold: bbox[2] + self.crop_threshold]
        #     cv2.imwrite(rf'logs/arrows/arrow_{time.time()}.jpg', crop_img)

    def get_all_elements(self):
        return self.texts + self.arrows + self.inputs + self.decisions + self.knowledge_model + self.knowledge_source

    def split_elements(self, elements):
        all_types = {
            ElementsType.NODE_INPUT_DATA.value: [],
            ElementsType.TEXT.value: [],
            ElementsType.NODE_DECISION.value: [],
            ElementsType.ARROW.value: [],
            ElementsType.NODE_KNOWLEDGE_SOURCE.value: [],
            ElementsType.NODE_KNOWLEDGE_MODEL.value: []
        }
        for element in elements:
            if element.id.startswith('arrow'):
                arrow = element.id.split('_')[0]
                if arrow in all_types:
                    all_types[arrow].append(element)
            else:
                elem_type, _, _ = element.id.rpartition('_')
                if elem_type in all_types:
                    all_types[elem_type].append(element)

        return all_types[ElementsType.TEXT.value], all_types[ElementsType.ARROW.value], \
               all_types[ElementsType.NODE_INPUT_DATA.value], all_types[ElementsType.NODE_DECISION.value], \
               all_types[ElementsType.NODE_KNOWLEDGE_MODEL.value], all_types[ElementsType.NODE_KNOWLEDGE_SOURCE.value],

    def create_output(self, output):
        make_dir(output)
        return output / f"{self.name}.xml"

    @staticmethod
    def is_first_area_bigger(bbox1, bbox2):
        """
        Bboxes have to be a tuple in order: x min, y min, x max, y max
        """
        x_min1, y_min1, x_max1, y_max1 = bbox1
        x_min2, y_min2, x_max2, y_max2 = bbox2
        bbox2_area = (x_max2 - x_min2) * (y_max2 - y_min2)
        bbox1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
        if bbox1_area > bbox2_area:
            return True
        return False

    @staticmethod
    def is_bbox1_inside_bbox2(bbox1, bbox2):
        """
        Bboxes have to be a tuple in order: x min, y min, x max, y max
        """
        x_min1, y_min1, x_max1, y_max1 = bbox1
        x_min2, y_min2, x_max2, y_max2 = bbox2

        if x_min1 >= x_min2 and y_min1 >= y_min2 and x_max1 <= x_max2 and y_max1 <= y_max2:
            return True
        return False

    def bound_text_elements(self):
        bounded_text_elements_counter = 0
        categories = [self.decisions, self.inputs, self.knowledge_model, self.knowledge_source]
        for text in self.texts:
            for category in categories:
                for element in category:
                    if self.is_bbox1_inside_bbox2(text.bbox, element.bbox):
                        bounded_text_elements_counter += 1
                        element.name = text.name
                        break
        if len(self.texts) == bounded_text_elements_counter:
            print("ALL TEXTS FITTED")
            self.texts = []
        if bounded_text_elements_counter < len(self.texts):
            raise ValueError(f"Not all text elements are bound to an element in {categories}")

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

        # Add a variable to keep track of the overall nearest element
        nearest_element = None
        smallest_distance = np.inf

        for element in elements:
            # Calculate the center of the element's bbox
            center = ((element.bbox[0] + element.bbox[2]) / 2,
                      (element.bbox[1] + element.bbox[3]) / 2)

            # Calculate the vector from the point to the element
            vector = (center[0] - point[0], center[1] - point[1])

            # Calculate the angle to the element
            angle = np.arccos(np.dot(direction, vector) /
                              (np.linalg.norm(direction) * np.linalg.norm(vector)))

            # Calculate the distance to the element
            distance = np.linalg.norm(vector)

            # Keep track of the nearest element overall, regardless of direction
            if distance < smallest_distance:
                smallest_distance = distance
                nearest_element = element

            # If the angle is too large, ignore the element
            if np.abs(angle) > np.pi / 2:
                continue

            # Update the nearest element if this element is closer in the direction
            if vector[0] < 0 and distance < smallest_distance_start:
                smallest_distance_start = distance
                nearest_element_start = element
            elif vector[0] > 0 and distance < smallest_distance_end:
                smallest_distance_end = distance
                nearest_element_end = element

            # If no elements were found in the specified direction, use the overall nearest element
        if nearest_element_start is None:
            nearest_element_start = nearest_element
        if nearest_element_end is None:
            nearest_element_end = nearest_element

        return nearest_element_start.id, nearest_element_end.id

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

    def create_xml(self):
        definitions = ET.Element("definitions", attrib={"xmlns": self.xmlns[""], "xmlns:dmndi": self.xmlns["dmndi"]})
        dmndi = ET.SubElement(definitions, "dmndi:DMNDI")
        dmnDiagram = ET.SubElement(dmndi, "dmndi:DMNDiagram", attrib={"id": "DMNDiagram_0"})

        elements_map = {}
        dmn_shape_ids = set()

        for element in self.decisions:
            decision = ET.SubElement(definitions, "decision", attrib={"id": element._id, "name": element.name})
            elements_map[element._id] = {"xml_element": decision, "bbox": element.bbox}

        for element in self.inputs:
            input_data = ET.SubElement(definitions, "inputData", attrib={"id": element._id, "name": element.name})
            elements_map[element._id] = {"xml_element": input_data, "bbox": element.bbox}

        for element in self.knowledge_model:
            knowledge_model = ET.SubElement(definitions, "businessKnowledgeModel", attrib={"id": element._id, "name": element.name})
            elements_map[element._id] = {"xml_element": knowledge_model, "bbox": element.bbox}

        for element in self.knowledge_source:
            knowledge_source = ET.SubElement(definitions, "knowledgeSource", attrib={"id": element._id, "name": element.name})
            elements_map[element._id] = {"xml_element": knowledge_source, "bbox": element.bbox}

        for element in self.arrows:
            if element.source in elements_map and element.target in elements_map:
                source_element = elements_map[element.source]["xml_element"]
                source_bbox = elements_map[element.source]["bbox"]

                target_element = elements_map[element.target]["xml_element"]
                target_bbox = elements_map[element.target]["bbox"]

                if isinstance(target_element, ET.Element):
                    if target_element.tag.endswith("decision"):
                        infoReq = ET.SubElement(target_element, "informationRequirement", attrib={"id": element._id})

                        if isinstance(source_element, ET.Element):
                            if source_element.tag.endswith("inputData"):
                                ET.SubElement(infoReq, "requiredInput",
                                              attrib={"href": "#" + source_element.attrib['id']})
                            elif source_element.tag.endswith("businessKnowledgeModel"):
                                ET.SubElement(infoReq, "requiredKnowledge",
                                              attrib={"href": "#" + source_element.attrib['id']})
                    elif target_element.tag.endswith("knowledgeSource"):
                        authReq = ET.SubElement(target_element, "authorityRequirement", attrib={"id": element._id})

                        if isinstance(source_element, ET.Element):
                            if source_element.tag.endswith("inputData"):
                                ET.SubElement(authReq, "requiredInput",
                                              attrib={"href": "#" + source_element.attrib['id']})

                    dmnEdge = ET.SubElement(dmnDiagram, "dmndi:DMNEdge",
                                            attrib={"id": "DMNEdge_" + element._id,
                                                    "dmnElementRef": element._id})

                    self.add_waypoints(dmnEdge, source_bbox, target_bbox)

                    if element.source not in dmn_shape_ids:
                        self.add_dmn_shape(dmnDiagram, element.source, source_bbox)
                        dmn_shape_ids.add(element.source)

                    if element.target not in dmn_shape_ids:
                        self.add_dmn_shape(dmnDiagram, element.target, target_bbox)
                        dmn_shape_ids.add(element.target)

        xml_string = ET.tostring(definitions, encoding='unicode')

        xml_declaration = f'<?xml version="1.0" encoding="UTF-8"?>\n<definitions id="dmn_visualizer" name="{self.name}"'
        for prefix, uri in self.xmlns.items():
            if prefix:
                xml_declaration += f' xmlns:{prefix}="{uri}"'
            else:
                xml_declaration += f' xmlns="{uri}"'
        xml_declaration += '>\n'

        xml_string = xml_declaration + xml_string[xml_string.find("<dmndi:DMNDI>"):]

        dom = minidom.parseString(xml_string)
        pretty_xml = dom.toprettyxml()
        """
        WARNING: workaround needed to delete xmlns:dc="http://www.omg.org/spec/DMN/20191111/DC/" and 
        xmlns:di="http://www.omg.org/spec/DMN/20191111/DI/" from definitions after minidom.parseString(xml_String) 
        (needed there for xml.parsers.expat.ExpatError: no element found), now we can visualize via 
        dmn-js but still got warnings described at my issue list
        """
        pretty_xml = pretty_xml.replace('xmlns:dc="http://www.omg.org/spec/DMN/20191111/DC/" ', '')
        pretty_xml = pretty_xml.replace('xmlns:di="http://www.omg.org/spec/DMN/20191111/DI/"', '')

        with open(self.output_path, 'w') as f:
            print(f"Creating {self.output_path}")
            f.write(pretty_xml)

    def add_dmn_shape(self, parent, element_id, bbox):
        dmn_shape = ET.SubElement(parent, "dmndi:DMNShape", attrib={"id": "DMNShape_" + element_id, "dmnElementRef": element_id})
        x1, y1, x2, y2 = bbox
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        ET.SubElement(dmn_shape, "dc:Bounds", attrib={"x": str(x1), "y": str(y1), "width": str(width), "height": str(height)})

    # def add_waypoints(self, parent, source_bbox, target_bbox):
    #     sx1, sy1, sx2, sy2 = source_bbox
    #     tx1, ty1, tx2, ty2 = target_bbox
    #
    #     source_center = ((sx1 + sx2) / 2, (sy1 + sy2) / 2)
    #     target_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
    #
    #     if source_center[1] < target_center[1]:  # If the source is above the target
    #         source_point = (source_center[0], sy2)
    #         target_point = (target_center[0], ty1)
    #     elif source_center[1] > target_center[1]:  # If the source is below the target
    #         source_point = (source_center[0], sy1)
    #         target_point = (target_center[0], ty2)
    #     elif source_center[0] < target_center[0]:  # If the source is to the left of the target
    #         source_point = (sx2, source_center[1])
    #         target_point = (tx1, target_center[1])
    #     else:  # If the source is to the right of the target
    #         source_point = (sx1, source_center[1])
    #         target_point = (tx2, target_center[1])
    #
    #     ET.SubElement(parent, "di:waypoint", attrib={"x": str(source_point[0]), "y": str(source_point[1])})
    #     ET.SubElement(parent, "di:waypoint", attrib={"x": str(target_point[0]), "y": str(target_point[1])})

    def add_waypoints(self, parent, source_bbox, target_bbox):
        sx1, sy1, sx2, sy2 = source_bbox
        tx1, ty1, tx2, ty2 = target_bbox

        source_center = ((sx1 + sx2) / 2, (sy1 + sy2) / 2)
        target_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)

        x_distance = abs(source_center[0] - target_center[0])
        y_distance = abs(source_center[1] - target_center[1])

        if y_distance >= x_distance:
            if source_center[1] < target_center[1]:  # If the source is above the target
                source_point = (source_center[0], sy2)
                target_point = (target_center[0], ty1)
            else:  # If the source is below the target
                source_point = (source_center[0], sy1)
                target_point = (target_center[0], ty2)
        else:
            if source_center[0] < target_center[0]:  # If the source is to the left of the target
                source_point = (sx2, source_center[1])
                target_point = (tx1, target_center[1])
            else:  # If the source is to the right of the target
                source_point = (sx1, source_center[1])
                target_point = (tx2, target_center[1])

        ET.SubElement(parent, "di:waypoint", attrib={"x": str(source_point[0]), "y": str(source_point[1])})
        ET.SubElement(parent, "di:waypoint", attrib={"x": str(target_point[0]), "y": str(target_point[1])})
