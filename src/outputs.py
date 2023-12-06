import os
import numpy as np 
from scipy.io import savemat

def create_output(output, output_file_path):
    # Verificar se o diretório de destino existe, se não, criar
    output_directory = os.path.dirname(output_file_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Escrever a saída no arquivo
    with open(output_file_path, 'w') as output_file:
        for row in output:
            output_file.write('\t'.join(map(str, row)) + '\n')
    print(f'Saída salva em: {output_file_path}')

    
def create_output_keypoints(kp_list, output_file_path, nr_points):
    # Verificar se o diretório de destino existe, se não, criar
    output_directory = os.path.dirname(output_file_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    




    for item in kp_list:
        savemat(output_file_path, {'data': item})
        break
    # Escrever a saída no arquivo
   # with open(output_file_path, 'w') as output_file:
      #  for keypoint in kp_list:
          #     for i in range(keypoint.shape[1]):
            #       x, y = str(keypoint[0,i]) , str(keypoint[1,i])
             #      descriptor = keypoint[2:,i]
             #      descriptors_str = "\t".join(str(d) for d in descriptor) 
             #      output_file.write(f"{x}\t{y}\t{descriptors_str}\n")
                
                
    #print(f'Saída salva em: {output_file_path}')