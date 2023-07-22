from classes import GraphImage
from pathlib import Path
import argparse


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
                        default=r'C:\Users\ttwoj\PycharmProjects\dmn_builder\TensorFlow\research\object_detection'
                                r'\my_first_model\saved_model')
    parser.add_argument('--img_file', '-f', help='Path to image', type=Path)
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--output', '-o', help='Output directory.', type=Path, required=False)
    parser.add_argument('--xml', '-x', help='Path to xml file', type=Path, required=False)
    return parser.parse_args()


def main():
    args = parse_args()
    img = r'C:\Users\ttwoj\PycharmProjects\local_dmn_visualizer\data\dmn_ex01_001.jpeg'
    verbose = True
    label_map = r'model/label_map.pbtxt'
    model = r'model/saved_model'
    graph = GraphImage(img, model, label_map, verbose=verbose)
    elements = graph.parse_elements()

    for e in elements:
        print(e)


if __name__ == "__main__":
    main()
