import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from utils import make_dir
from collections import defaultdict


class XMLCreator:
    def __init__(self, elements, name, output_path):
        self.elements = elements
        self.name = name
        self.output_path = self.create_output(output_path)
        self.xmlns = {"": "https://www.omg.org/spec/DMN/20191111/MODEL/",
                      "dmndi": "https://www.omg.org/spec/DMN/20191111/DMNDI/",
                      "dc": "http://www.omg.org/spec/DMN/20191111/DC/",
                      "di": "http://www.omg.org/spec/DMN/20191111/DI/"}

        ET.register_namespace("", self.xmlns[""])
        ET.register_namespace("dmndi", self.xmlns["dmndi"])
        ET.register_namespace("dc", self.xmlns["dc"])
        ET.register_namespace("di", self.xmlns["di"])

    def create_output(self, output):
        make_dir(output)
        return output / f"{self.name}.xml"

    def create_xml(self):
        definitions = ET.Element("definitions", attrib={"xmlns": self.xmlns[""],
                                                        "xmlns:dmndi": self.xmlns["dmndi"]})
        dmndi = ET.SubElement(definitions, "dmndi:DMNDI")
        dmnDiagram = ET.SubElement(dmndi, "dmndi:DMNDiagram", attrib={"id": "DMNDiagram_0"})

        decisions = []
        inputs = []
        arrows = []

        for element in self.elements:
            if "decision" in element.id:
                decision = ET.SubElement(definitions, "decision", attrib={"id": element.id, "name": element.name})
                decisions.append({"xml_element": decision, "element": element})

            if "input_data" in element.id:
                inputData = ET.SubElement(definitions, "inputData", attrib={"id": element.id, "name": element.name})
                inputs.append({"xml_element": inputData, "element": element})

            if "arrow" in element.id:
                arrows.append(element)

            ET.SubElement(ET.SubElement(dmnDiagram, "dmndi:DMNShape",
                                        attrib={"id": "DMNShape_" + element.id,
                                                "dmnElementRef": element.id}),
                          "dc:Bounds",
                          attrib={"height": str(element.bbox[3] - element.bbox[1]),
                                  "width": str(element.bbox[2] - element.bbox[0]),
                                  "x": str(element.bbox[0]),
                                  "y": str(element.bbox[1])})
        # debug_xml_string = ET.tostring(definitions, encoding='unicode')
        # print(debug_xml_string)
        # print()
        print("Works fine to here")
        # we need DMNEdges information requirenments

        # first block
        for arrow in arrows:
            for decision in decisions:
                if self.check_intersection(arrow.bbox, decision["element"].bbox):
                    infoReq = ET.SubElement(decision["xml_element"], "informationRequirement",
                                            attrib={"id": "InformationRequirement_" + arrow.id})
                    ET.SubElement(infoReq, "requiredDecision", attrib={"href": "#" + decision["element"].id})

                    dmnEdge = ET.SubElement(dmnDiagram, "dmndi:DMNEdge", attrib={"id": "DMNEdge_" + arrow.id,
                                                                                 "dmnElementRef": "InformationRequirement_" + arrow.id})
                    self.add_waypoints(dmnEdge, arrow.bbox, decision["element"].bbox)
                # break
        debug_xml_string = ET.tostring(definitions, encoding='unicode')
        print(debug_xml_string)
        print()
        # second block
        for input_data in inputs:
            if self.check_intersection(arrow.bbox, input_data["element"].bbox):
                infoReq = ET.SubElement(decision["xml_element"], "informationRequirement",
                                        attrib={"id": "InformationRequirement_" + arrow.id})
                ET.SubElement(infoReq, "requiredInput", attrib={"href": "#" + input_data["element"].id})

                dmnEdge = ET.SubElement(dmnDiagram, "dmndi:DMNEdge", attrib={"id": "DMNEdge_" + arrow.id,
                                                                             "dmnElementRef": "InformationRequirement_" + arrow.id})
                self.add_waypoints(dmnEdge, arrow.bbox, input_data["element"].bbox)

        # Convert ElementTree to string

        print(definitions)
        xml_string = ET.tostring(definitions, encoding='unicode')
        print()
        print(xml_string)
        # Construct XML declaration from namespaces
        xml_declaration = f'<?xml version="1.0" encoding="UTF-8"?>\n<definitions id="dmn_visualizer" name="{self.name}"'
        for prefix, uri in self.xmlns.items():
            if prefix:
                xml_declaration += f' xmlns:{prefix}="{uri}"'
            else:
                xml_declaration += f' xmlns="{uri}"'
        xml_declaration += '>\n'
        print()
        print(xml_declaration)
        xml_string = xml_declaration + xml_string[xml_string.find("<dmndi:DMNDI>"):]

        # Parse string to an XML document and create a pretty (indented) version
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

        # Save to file
        with open(self.output_path, 'w') as f:
            f.write(pretty_xml)

    @staticmethod
    def check_intersection(arrow_bbox, element_bbox):
        ax1, ay1, ax2, ay2 = arrow_bbox
        ex1, ey1, ex2, ey2 = element_bbox

        return ax1 < ex2 and ax2 > ex1 and ay1 < ey2 and ay2 > ey1

    def add_waypoints(self, parent, arrow_bbox, element_bbox):
        ax1, ay1, ax2, ay2 = arrow_bbox
        ex1, ey1, ex2, ey2 = element_bbox

        arrow_center = ((ax1 + ax2) / 2, (ay1 + ay2) / 2)
        element_center = ((ex1 + ex2) / 2, (ey1 + ey2) / 2)

        ET.SubElement(parent, "di:waypoint", attrib={"x": str(arrow_center[0]), "y": str(arrow_center[1])})
        ET.SubElement(parent, "di:waypoint", attrib={"x": str(element_center[0]), "y": str(element_center[1])})
