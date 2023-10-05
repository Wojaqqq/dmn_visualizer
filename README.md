# Dmn Visualizer

This project is developed as part of a Master's thesis and aims to create a tool that can convert DMN (Decision Model and Notation) diagrams from scans or photos into XML format. 


## Introduction

The tool is engineered to analyze images containing Decision Model and Notation (DMN) symbols and subsequently generate corresponding XML files. These XML files can be rendered and interacted with through the use of the [dmn-js](https://bpmn.io/toolkit/dmn-js/) visualization tool. The project encompasses a neural network that has been trained on a dataset, the details of which are elaborated in the "related" chapter. Additionally, the system incorporates an algorithm specifically designed to produce accurate and reliable output. To achieve this, the tool employs a combination of object detection algorithms and Handwritten Text Recognition technologies. 



## Setup and Installation

1. Clone the repository.
2. Navigate to the project directory.
3. Create a new Anaconda environment:

```bash
conda env create -f conda_environment.yaml
```
4. Download the trained model from [this link](https://drive.google.com/drive/folders/1DfO4WJb3h8rCiO8LThvFJkIOEWHDBRKz?usp=sharing) and move it to the `model` directory.



## Usage


### Command line arguments:
* `--mode`: specifies the operational mode, which can be either "detection" or "xml". The default value is "detection".
* `--model`: specifies the path to the machine learning model. The default path is "model/saved_model".
* `--img_file`: specifies the required path to the input image file.
* `--verbose`: a flag that, when set, enables the logging mechanism and additional output messages.
* `--xml`: used in "xml" mode to specify the path to the input XML file.
* `--output`: specifies the directory where output files will be saved. If not provided, a default output directory will be created.


### Examples
Use help function to see additional information.
```bash
python dmn_visualizer.py --help
```

As example run tool on *test_image.jpg*.
```bash
python dmn_visualizer.py --img_file path/to/test_image.jpg --output path/to/output --verbose
```

## Related

* [Dataset with annotations](https://github.com/Wojaqqq/hdDMN)
* [HTR model used in project](https://github.com/githubharald/SimpleHTR)
