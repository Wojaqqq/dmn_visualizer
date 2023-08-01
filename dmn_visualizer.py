from base_classes import GraphImage
from xml_creator import XMLCreator
from pathlib import Path
import argparse
import glob
import re


class CustomHelpFormatter(argparse.HelpFormatter):
    def format_help(self):
        ascii_art = """
          _____  __  __ _   _  __      ___                 _ _              
         |  __ \|  \/  | \ | | \ \    / (_)               | (_)             
         | |  | | \  / |  \| |  \ \  / / _ ___ _   _  __ _| |_ _______ _ __ 
         | |  | | |\/| | . ` |   \ \/ / | / __| | | |/ _` | | |_  / _ \ '__|
         | |__| | |  | | |\  |    \  /  | \__ \ |_| | (_| | | |/ /  __/ |   
         |_____/|_|  |_|_| \_|     \/   |_|___/\__,_|\__ ,_|_|_/___\___|_|   
        """
        original_help = super().format_help()
        return "{}\n{}".format(ascii_art, original_help)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    parser.add_argument('--mode', '-d', help='Base source of information.', choices=['detection', 'xml'],
                        default='detection')
    parser.add_argument('--model', '-m', help='Model used for detection.', type=Path,
                        default=r'model/saved_model')
    parser.add_argument('--img_file', '-f', help='Path to image', type=Path)
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--output', '-o', help='Output directory.', type=Path, required=False)
    parser.add_argument('--xml', '-x', help='Path to xml file', type=Path, required=False)
    return parser.parse_args()


def dmn_visualizer():
    args = parse_args()


def run_on_directory():
    working_dir = Path('D:\Biblioteki\Puplit\Magisterka\imgs')
    for img_path in glob.glob(fr'{working_dir}/*.jpeg'):
        img_path = Path(img_path)
        verbose = False
        label_map = r'model/label_map.pbtxt'
        model = r'model/saved_model'
        graph = GraphImage(Path(img_path), model, label_map, verbose=verbose)
        elements = graph.parse_elements()
        output = Path('output')
        creator = XMLCreator(elements, f"out_31.07_{Path(img_path).stem}", output, img_path)
        creator.bound_text_elements()
        creator.bound_arrow_elements()

        output = 'logs/07_31_23_test'
        with open(f'{output}/{Path(img_path).stem}.txt', 'w') as f:
            for e in creator.get_all_elements():
                f.write(e.__str__() + '\n')

        creator.create_xml()


def run_on_single_image():
    img_path = Path(r'data/dmn_ex04_002.jpeg')
    verbose = False
    label_map = r'model/label_map.pbtxt'
    model = r'model/saved_model'
    graph = GraphImage(img_path, model, label_map, verbose=verbose)
    elements = graph.parse_elements()
    output = Path('output')
    output_xml_name = img_path.stem

    for e in elements:
        print(e)

    creator = XMLCreator(elements, output_xml_name, output, img_path)
    print("AFTER XMLCreator creation")
    for e in creator.get_all_elements():
        print(e)

    creator.bound_text_elements()
    print("AFTER BOUNDING TEXT ELEMENTS")
    for e in creator.get_all_elements():
        print(e)

    creator.bound_arrow_elements()
    print("AFTER BOUNDING ARROW ELEMENTS")
    for e in creator.get_all_elements():
        print(e)

    creator.create_xml()


def run_on_fixed_data():
    path_to_set_txt_file = Path(r'..\working_elements_sets\ex01_001_set.txt')
    output = Path('output')
    output_xml_name = f'new_{path_to_set_txt_file.stem}'
    img_path = Path(r'data/dmn_ex01_001.jpeg')

    class TestElement:
        def __init__(self, _id, name, bbox, img, source=None, target=None):
            self._id = _id
            self.bbox = bbox
            self.img_path = img
            self.crop_threshold = 10
            self.save_crop_img = True
            self.source = source
            self.target = target
            self.name = name

        @property
        def id(self):
            return self._id

        def __str__(self):
            if self.source is None and self.target is None:
                return f"Element(Id: {self._id}, Name: {self.name}, Bbox: {self.bbox})"
            return f"Element(Id: {self._id}, Name: {self.name}, Bbox: {self.bbox}, Source: {self.source}, Target: {self.target})"

    def parse_element(line):
        regex = r"Element\(Id: (.*?), Name: (.*?), Bbox: \((\d+), (\d+), (\d+), (\d+)\)(, Source: (.*?), Target: (.*?))?\)"
        match = re.match(regex, line)

        _id = match.group(1)
        name = match.group(2)
        bbox = (int(match.group(3)), int(match.group(4)), int(match.group(5)), int(match.group(6)))
        img = "path_to_img"  # placeholder, replace with actual logic
        source = match.group(8) if match.group(8) else None
        target = match.group(9) if match.group(9) else None

        return TestElement(_id, name, bbox, img, source, target)

    elements = []

    with open(path_to_set_txt_file, 'r') as file:
        for line in file:
            elements.append(parse_element(line))

    for e in elements:
        print(e)

    creator = XMLCreator(elements, output_xml_name, output, img_path)
    creator.create_xml()


if __name__ == "__main__":
    # run_on_fixed_data()
    # run_on_single_image()
    run_on_directory()
