import subprocess
import os
import logging
import re
from base_classes import TestElement
logger = logging.getLogger()


def make_dir(path):
    if not os.path.exists(path) and path is not None:
        os.makedirs(path)


def load_label_map(label_map_path):
    lbl_map = {}
    with open(label_map_path, 'r') as f:
        for line in f:
            if 'id:' in line:
                label_id = int(line.strip().split(':')[-1])
            elif 'name:' in line:
                label_name = line.strip().split(':')[-1].strip().replace("'", "")
                lbl_map[label_id] = label_name
    return lbl_map


def simpleHTR_word_detector(img):
    """
    Running through a subprocess of the modified handwritten word detection script from the https://github.com/githubharald/SimpleHTR.git
    """
    process = subprocess.run(['python', r'SimpleHTR\src\main.py', '--img_file', img],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
    logger.info(process.stderr.strip().split('\n')[-1])
    output = process.stdout.strip().split(';')
    if len(output) != 2:
        return None, 0
    return output[0], output[1]


def parse_xml_element(line):
    regex = r"Element\(Id: (.*?), Name: (.*?), Bbox: \((\d+), (\d+), (\d+), (\d+)\)(, Source: (.*?), Target: (.*?))?\)"
    match = re.match(regex, line)

    _id = match.group(1)
    name = match.group(2)
    bbox = (int(match.group(3)), int(match.group(4)), int(match.group(5)), int(match.group(6)))
    img = "path_to_img"
    source = match.group(8) if match.group(8) else None
    target = match.group(9) if match.group(9) else None

    return TestElement(_id, name, bbox, img, source, target)