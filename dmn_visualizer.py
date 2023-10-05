import argparse
import time
import logging

from base_classes import GraphImage
from xml_creator import XMLCreator
from pathlib import Path
from utils import make_dir

logger = logging.getLogger()


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
    parser.add_argument('--img_file', '-f', help='Path to image', type=Path, required=True)
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--output', '-o', help='Output directory.', type=Path, default=Path())
    parser.add_argument('--xml', '-x', help='Path to xml file', type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    label_map = r'model/label_map.pbtxt'
    if args.verbose:
        make_dir('logs')
        logging.basicConfig(filename=f"logs/dmn_visualizer_{int(time.time())}.log",
                            format='[%(asctime)s %(filename)s] %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            filemode='w',
                            level='INFO')
    else:
        logging.basicConfig(level=logging.CRITICAL)

    print(f"Launching dmn_visualizer for {args.img_file}")
    logger.info(f"General info -> img_file: {args.img_file}, verbose: {args.verbose}, label_map: {label_map}, "
                f"model: {args.model}, output: {args.output}")
    graph = GraphImage(args.img_file, args.model, label_map, verbose=args.verbose)
    elements = graph.parse_elements()

    for e in elements:
        logger.info(f"{str(e)}")

    creator = XMLCreator(elements, args.img_file.stem, args.output, args.img_file)
    logger.info("Elements after XML Creation")
    for e in creator.get_all_elements():
        logger.info(f"{str(e)}")

    creator.bound_text_elements()
    logger.info("Elements after bounding text")
    for e in creator.get_all_elements():
        logger.info(f"{str(e)}")

    creator.bound_arrow_elements()
    logger.info("Elements after bounding arrows")
    for e in creator.get_all_elements():
        logger.info(f"{str(e)}")

    creator.create_xml()
