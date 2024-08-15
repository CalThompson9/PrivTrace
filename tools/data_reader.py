import numpy as np
import os
import re
import config.folder_and_file_names as config


class DataReader:

    def __init__(self):
        pass

    def read_trajectories_from_data_file(self, file_n):
        file_name = os.path.join('.', config.trajectory_data_folder, file_n)
        trajectory_list = self.read_tra_data(file_name)
        return trajectory_list

    def read_tra_data(self, file_name):
        trajectory_list = []

        try:
            with open(file_name, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        trajectory_data_carrier = line[3:]
                        trajectory_data_list = re.split('[,;]+', trajectory_data_carrier.strip())[:-1]
                        trajectory_data_list = list(map(float, trajectory_data_list))
                        
                        # If the list length is odd, remove the last element to make it even
                        if len(trajectory_data_list) % 2 != 0:
                            print(f"Odd number of elements found, removing the last element: {trajectory_data_list[-1]}")
                            trajectory_data_list.pop()
                        
                        trajectory_array = np.array(trajectory_data_list).reshape((-1, 2))
                        trajectory_list.append(trajectory_array)
        except Exception as e:
            print(f"Error reading data from file: {e}")
            raise

        return trajectory_list
