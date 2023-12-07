import os
import numpy as np 
from scipy.io import savemat

def create_output(output, output_file_path):
    # Verificar se o diretório de destino existe, se não, criar
    output_directory = os.path.dirname(output_file_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    savemat(output_file_path, {'Homographies': output}) #save the outputs into a .mat file 

    
def create_output_keypoints(kp_list, output_file_path, nr_points):
    # Verificar se o diretório de destino existe, se não, criar
    output_directory = os.path.dirname(output_file_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    new_list = []
    for item in kp_list:
        points = item[:,:nr_points]  #put the size of each array in an uniform way
        points = points.reshape((points.shape[1],points.shape[0]))
        new_list.append(points)

    savemat(output_file_path, {'Keypoints': new_list})

  
