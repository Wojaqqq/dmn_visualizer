import subprocess
import os
import time
from pathlib import Path


def make_dir(path):
    if not os.path.exists(path):
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
    make_dir('logs')
    # TODO change not include img
    log_file_path = os.path.join('logs', f'word_detector_log_{time.time()}_{Path(img).stem}.txt')
    with open(log_file_path, 'a') as log_file:
        process = subprocess.run(['python', r'SimpleHTR\src\main.py', '--img_file', img],
                                 stdout=subprocess.PIPE,
                                 stderr=log_file,
                                 universal_newlines=True)
    output = process.stdout.strip().split(';')
    if len(output) != 2:
        return None, 0
    return output[0], output[1]
