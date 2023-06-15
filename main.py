from classes import XMLParser

xml_file = r'C:\Users\ttwoj\PycharmProjects\dmn_visualizer\dmn_ex01_005.xml'
img = r'C:\Users\ttwoj\PycharmProjects\dmn_visualizer\dmn_ex01_005.jpeg'


def main():

    xml_parser = XMLParser(xml_file, img)

    elements = xml_parser.parse_elements()
    print(elements)
    # presenter = GraphPresenter(elements)
    # presenter.generate_presentation()


if __name__ == "__main__":
    main()
