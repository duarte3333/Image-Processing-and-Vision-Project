import os

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

    
def create_output_keypoints(kp_list, output_file_path):
    # Verificar se o diretório de destino existe, se não, criar
    output_directory = os.path.dirname(output_file_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Escrever a saída no arquivo
    with open(output_file_path, 'w') as output_file:
        for frame_number, keypoints in enumerate(kp_list, start=1):
            for keypoint in kp_list:
                    x, y = keypoint[0] , keypoint[1]
                    descriptor = keypoint[2:]
                    descriptors_str = "\t".join(str(d) for d in descriptor)

                    output_file.write(f"{x}\t{y}\t{descriptors_str}\n")
    print(f'Saída salva em: {output_file_path}')