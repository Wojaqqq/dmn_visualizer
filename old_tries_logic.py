import xml.etree.ElementTree as ET
import pygraphviz as pgv

file_path = r"/home/tomasz/Projects/mgr/dmn_visualizer/to_annotate_1.xml"


def parse_xml(path_to_file):
    tree = ET.parse(file_path)
    root = tree.getroot()

    objects = root.findall('.//object')
    object_dict = {}

    for obj in objects:
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

    res = []
    for key, values in object_dict.items():
        for i, value in enumerate(values):
            res.append((f"{key}_{i}", value))

    print(res)
    return res


result = parse_xml(file_path)

result = [
    ('node_decision_0', (1301, 770, 1994, 1129)),
    ('node_decision_1', (986, 285, 1592, 580)),
    ('node_input_data_0', (1099, 1442, 1633, 1587)),
    ('node_knowledge_source_0', (171, 349, 505, 595)),
    ('node_knowledge_source_1', (2375, 857, 2824, 1132)),
    ('node_knowledge_model_0', (2007, 1385, 2554, 1627)),
    ('arrow_information_requirement_0', (1354, 570, 1552, 772)),
    ('arrow_information_requirement_1', (1364, 1125, 1556, 1444)),
    ('arrow_knowledge_requirement_0', (1862, 1115, 2182, 1406)),
    ('arrow_authority_requirement_0', (1984, 902, 2386, 1006)),
    ('arrow_authority_requirement_1', (499, 387, 988, 491)),
    ('text_0', (1024, 357, 1522, 487)),
    ('text_1', (235, 408, 439, 525)),
    ('text_2', (1369, 857, 1845, 1023)),
    ('text_3', (2401, 891, 2771, 1008)),
    ('text_4', (2143, 1438, 2390, 1551)),
    ('text_5', (1164, 1478, 1558, 1559))
]


def filter_objects(name_prefix, objects):
    return [(name, bbox) for name, bbox in objects if name.startswith(name_prefix)]


def find_label(node_bbox, text_objects):
    for text_name, text_bbox in text_objects:
        xmin, ymin, xmax, ymax = node_bbox
        x_center, y_center = (xmin + xmax) // 2, (ymin + ymax) // 2
        if text_bbox[0] <= x_center <= text_bbox[2] and text_bbox[1] <= y_center <= text_bbox[3]:
            return text_name  # Return the full text_name instead of just the text part
    return None


# Filter objects based on their names
node_objects = filter_objects("node", result)
text_objects = filter_objects("text", result)
arrow_objects = filter_objects("arrow", result)

# Create a PyGraphviz graph
G = pgv.AGraph(directed=True)

# Add nodes to the graph
for name, bbox in node_objects:
    label = find_label(bbox, text_objects)
    G.add_node(name, label=label, shape='rectangle', bbox=bbox)

# # Add edges to the graph based on arrow objects  OLD VERSION
# for arrow_name, arrow_bbox in arrow_objects:
#     source_node = None
#     dest_node = None
#
#     for node_name, node_bbox in node_objects:
#         if arrow_bbox[0] >= node_bbox[2] and arrow_bbox[2] <= node_bbox[0]:
#             continue
#         if arrow_bbox[0] <= node_bbox[2] and arrow_bbox[2] >= node_bbox[0]:
#             if arrow_bbox[1] <= node_bbox[3] and arrow_bbox[3] >= node_bbox[1]:
#                 source_node = node_name
#             elif arrow_bbox[1] >= node_bbox[3] and arrow_bbox[3] <= node_bbox[1]:
#                 dest_node = node_name
#
#     if source_node and dest_node:
#         G.add_edge(source_node, dest_node)

# # Add edges to the graph based on arrow objects SECOND OLD
# for arrow_name, arrow_bbox in arrow_objects:
#     source_node = None
#     dest_node = None
#
#     axmin, aymin, axmax, aymax = arrow_bbox
#     arrow_center_x = (axmin + axmax) // 2
#     arrow_center_y = (aymin + aymax) // 2
#
#     for node_name, node_bbox in node_objects:
#         nxmin, nymin, nxmax, nymax = node_bbox
#
#         if arrow_center_x >= nxmin and arrow_center_x <= nxmax:
#             if arrow_center_y <= nymax and aymax >= nymax:
#                 source_node = node_name
#             elif arrow_center_y >= nymin and aymin <= nymin:
#                 dest_node = node_name
#
#     if source_node and dest_node:
#         G.add_edge(source_node, dest_node)

# # Add edges to the graph based on arrow objects THIRD OLD
# for arrow_name, arrow_bbox in arrow_objects:
#     source_node = None
#     dest_node = None
#
#     axmin, aymin, axmax, aymax = arrow_bbox
#
#     for node_name, node_bbox in node_objects:
#         nxmin, nymin, nxmax, nymax = node_bbox
#
#         if nymin <= aymin and nymax >= aymax:
#             if axmin >= nxmax:
#                 if not source_node or nxmax > source_node[1][2]:
#                     source_node = (node_name, node_bbox)
#             elif axmax <= nxmin:
#                 if not dest_node or nxmin < dest_node[1][0]:
#                     dest_node = (node_name, node_bbox)
#
#     if source_node and dest_node:
#         G.add_edge(source_node[0], dest_node[0])

def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

# Add edges to the graph based on arrow objects
for arrow_name, arrow_bbox in arrow_objects:
    source_node = None
    dest_node = None
    min_source_distance = float("inf")
    min_dest_distance = float("inf")

    arrow_start = (arrow_bbox[0], (arrow_bbox[1] + arrow_bbox[3]) // 2)
    arrow_end = (arrow_bbox[2], (arrow_bbox[1] + arrow_bbox[3]) // 2)

    for node_name, node_bbox in node_objects:
        nxmin, nymin, nxmax, nymax = node_bbox
        node_left = (nxmin, (nymin + nymax) // 2)
        node_right = (nxmax, (nymin + nymax) // 2)

        source_distance = distance(arrow_start, node_right)
        dest_distance = distance(arrow_end, node_left)

        if source_distance < min_source_distance:
            source_node = (node_name, node_bbox)
            min_source_distance = source_distance

        if dest_distance < min_dest_distance:
            dest_node = (node_name, node_bbox)
            min_dest_distance = dest_distance

    if source_node and dest_node:
        G.add_edge(source_node[0], dest_node[0])

# Render the graph to a PNG file
G.layout(prog='dot')
G.draw("graph.png", format='png')

