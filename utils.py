import subprocess


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
    Run via subprocess handwritten word detection from https://github.com/githubharald/SimpleHTR.git
    """
    # TODO add logs with sterr
    process = subprocess.run(['python', r'SimpleHTR\src\main.py', '--img_file', img],
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
    output = process.stdout.strip().split(';')
    return output[0], output[1]
