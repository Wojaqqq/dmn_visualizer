import xml.etree.ElementTree as ET
from PIL import Image
# import pygraphviz as pgv
import cv2
import pytesseract


class Element:
    def __init__(self, name, bbox, img):
        self.bbox = bbox
        self.img = img
        self.name = self.extract_text(name)

    def extract_text(self, name):
        if name.startswith('text'):
            image = cv2.imread(self.img)
            # print(f"\n\n{self.bbox}")
            bbox = list(self.bbox)
            crop_img = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            crop_img_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(crop_img_pil)
            return text
        else:
            return name


class XMLParser:
    def __init__(self, file_path, img):
        self.file_path = file_path
        self.root = self.load_xml()
        self.img = img

    def load_xml(self):
        tree = ET.parse(self.file_path)
        return tree.getroot()

    def parse_elements(self):
        # Assuming each 'object' in XML has 'bbox' and 'id'
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
        print(object_dict.items())
        for key, values in object_dict.items():
            for i, bbox in enumerate(values):
                # print(key)
                # print(values)
                # print(i)
                # print(bbox)
                elements.append(Element(f"{key}_{i}", bbox, self.img))

        return elements


# class GraphPresenter:
#     def __init__(self, elements):
#         self.elements = elements
#         self.graph = pgv.AGraph(directed=True)
#
#     def add_elements_to_graph(self):
#         for element in self.elements:
#             self.graph.add_node(element.id_, label=element.text)
#             # Here, you might also want to add edges between nodes depending on your requirements
#
#     def generate_presentation(self):
#         self.add_elements_to_graph()
#         self.graph.layout(prog='dot')
#         self.graph.draw('file.png')
