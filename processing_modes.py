import glob
import logging

from base_classes import GraphImage
from xml_creator import XMLCreator
from pathlib import Path
from utils import parse_xml_element

logger = logging.getLogger()


def run_on_directory():
    working_dir = Path('hdDMN\data\imgs')
    verbose = True
    label_map = r'model/label_map.pbtxt'
    model = r'model/saved_model'
    output = Path('output')
    logger.info(f"General info -> verbose: {verbose}, label_map: {label_map}, model: {model}, output: {output}")
    for img_path in glob.glob(fr'{working_dir}/*.jpeg'):
        img_path = Path(img_path)
        logger.info(f"IMAGE: {img_path.stem}")
        graph = GraphImage(Path(img_path), model, label_map, verbose=verbose)
        elements = graph.parse_elements()

        creator = XMLCreator(elements, f"out_31.07_{Path(img_path).stem}", Path(output), img_path)
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

        output = 'logs/092323'
        with open(f'{output}/{Path(img_path).stem}.txt', 'w') as f:
            for e in creator.get_all_elements():
                f.write(e.__str__() + '\n')

        creator.create_xml()
        logger.info("-" * 100)


def run_on_single_image():
    img_path = Path(r'data/dmn_ex05_022.jpeg')
    verbose = True
    label_map = r'model/label_map.pbtxt'
    model = r'model/saved_model'
    output = Path('output')
    output_xml_name = img_path.stem
    logger.info(
        f"General info -> img_path: {img_path}, verbose: {verbose}, label_map: {label_map}, model: {model}, output: {output}")
    graph = GraphImage(img_path, model, label_map, verbose=verbose)
    elements = graph.parse_elements()

    creator = XMLCreator(elements, output_xml_name, output, img_path)
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


def run_on_fixed_data():
    path_to_set_txt_file = Path(r'..\working_elements_sets\ex01_001_set.txt')
    output = Path('output')
    output_xml_name = f'original_{path_to_set_txt_file.stem}'
    img_path = Path(r'data/dmn_ex01_001.jpeg')
    elements = []

    with open(path_to_set_txt_file, 'r') as file:
        for line in file:
            elements.append(parse_xml_element(line))

    creator = XMLCreator(elements, output_xml_name, output, img_path)
    creator.create_xml()
