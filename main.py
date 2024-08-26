import config.folder_and_file_names as fname
from discretization.get_discretization import DisData
from primarkov.build_markov_model import ModelBuilder
from generator.state_trajectory_generation import StateGeneration
from generator.to_real_translator import RealLocationTranslator
from config.parameter_carrier import ParameterCarrier
from config.parameter_setter import ParSetter
from tools.data_writer import DataWriter
from data_preparation.data_preparer import DataPreparer
import os
import datetime

gridList = [4, 6, 8, 10, 12, 15, 20]
epsList = [0.05, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]

if __name__ == "__main__":
    writer = DataWriter()
    print('Begin all processes...')
    print(f'Start time: {datetime.datetime.now()}')

    par = ParSetter().set_up_args()
    pc = ParameterCarrier(par)

    # Create a directory for the dataset if one doesn't exist
    dataset_dir = f'./data/{pc.dataset_file_name.split(".")[0]}'
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Created directory: {dataset_dir}")

    for c in range(5):
        print(f"Processing fold {c}...")

        # Remove the .dat and any number on the end
        original_dataset_name = pc.dataset_file_name
        cleaned_dataset_name = ''.join(filter(str.isalpha, original_dataset_name.split('.')[0]))
        pc.dataset_file_name = f'{cleaned_dataset_name}{c}.dat'
        print(f"Cleaned dataset name: {pc.dataset_file_name} (was: {original_dataset_name})")

        par = ParSetter().set_up_args(dataset_file_name=pc.dataset_file_name)
        trajectory_set = DataPreparer(par).get_trajectory_set()
        print(f"Loaded trajectory set for {pc.dataset_file_name}")

        # Create a directory for each cv fold
        fold_dir = f'./data/{cleaned_dataset_name}/{c}'
        os.makedirs(fold_dir, exist_ok=True)
        print(f"Created directory for fold {c}: {fold_dir}")

        for epsilon in epsList:
            print(f"Processing epsilon {epsilon}...")
            pc.total_epsilon = epsilon

            disdata1 = DisData(pc)
            grid = disdata1.get_discrete_data(trajectory_set)
            print(f"Generated discrete data for epsilon {epsilon}")

            mb1 = ModelBuilder(pc)
            mo1 = mb1.build_model(grid, trajectory_set)
            print(f"Built Markov model for epsilon {epsilon}")

            mo1 = mb1.filter_model(trajectory_set, grid, mo1)
            print(f"Filtered Markov model for epsilon {epsilon}")

            sg1 = StateGeneration(pc)
            st_tra_list = sg1.generate_tra(mo1)
            print(f"Generated state trajectories for epsilon {epsilon}")

            rlt1 = RealLocationTranslator(pc)
            real_tra_list = rlt1.translate_trajectories(grid, st_tra_list)
            print(f"Translated state trajectories to real locations for epsilon {epsilon}")

            # Save the synthetic data
            result_file_path = f'{fold_dir}/{cleaned_dataset_name}-eps_{epsilon}.dat'
            writer.save_trajectory_data_in_list_to_file(real_tra_list, result_file_path)
            print(f"Saved synthetic data for epsilon {epsilon} at {result_file_path}")

    print('All processes completed.')
    print(f'End time: {datetime.datetime.now()}')
