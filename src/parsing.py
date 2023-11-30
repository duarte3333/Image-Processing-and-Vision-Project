import sys 
import os

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