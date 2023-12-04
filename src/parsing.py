import sys 
import os
import numpy as np

def is_valid_cfg_file(filename):
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower() == '.cfg'


def check_syntax():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        return 1
    if not os.path.exists(sys.argv[1]):
        print("File does not exist:", sys.argv[1])
        return 1
    if not is_valid_cfg_file(sys.argv[1]):
        print("Invalid file. Please provide a .cfg file.")
        return 1
    return 0

def parse_configuration_file(file_path):
    config_data = {'INPUT PARAMETERS': {}, 'OUTPUT PARAMETERS': {}}
    data = []
    with open(file_path, 'r') as file:
        current_section = None
        for line in file:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith("#") or not line:
                continue
            else:
                data.append(line)
    return data

def parse_points(config_data):
    """ Parse the points from the configuration file """
    line_map = config_data[1].split('  ')
    line_frame = config_data[2].split(' ')
    line_frame_og = [item for item in line_frame if item != ""]
    line_map = line_map[1:]
    line_frame_final = line_frame_og[2:]
    match_img = []
    match_map = []
    size = len(line_map)
    i = 0
    while (size > i):
        match_img.append((line_map[i].strip(), line_map[i+1].strip()))
        match_map.append((line_frame_final[i].strip(), line_frame_final[i+1].strip()))
        i += 2
    print("image matches: ", match_img, '\n')
    print("map matches: ", match_map, '\n')
    match_img = np.float32(match_img)
    match_map = np.float32(match_map)
    return match_img, match_map