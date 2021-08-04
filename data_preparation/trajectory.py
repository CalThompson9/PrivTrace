import numpy as np
from tools.general_tools import GeneralTools


class Trajectory:

    # define a numpy array traarr as the representation of a trajectory, in which sequences of rows are sequences of
    # points, the first and second column are longtitude and latitude. traarr is [-1, -1] in the beginning to
    # indicate it has not been initialized.
    def __init__(self):
        self.trajectory_array = np.array([-1, -1])
        # define index of  trajectory
        self.trajectory_index = -1
        # calculate number of  points in a trajctory
        self.point_number = 0
        # define a list to store index of cells and subcells that points are in in the trajectory
        self.trajectory_cell_list = np.array([], dtype=np.int)

        # self.trajectory_subcell_list = np.array([], dtype=np.int)

        # transform a trajectory from point sequence to grid sequence
        # self.grid_list = np.array([], dtype=int)
        # self.grid_sequence = np.array([], dtype=int)
        # self.grid_sequence_frequency = np.array([], dtype=int)
        self.level1_cell_index_sequence = np.array([], dtype=int)
        self.level2_cell_index_sequence = np.array([], dtype=int)

        self.cell_sequence = np.array([], dtype=int)
        self.cell_sequence_frequency = np.array([], dtype=int)

        self.usable_sequence = np.array([])
        self.usable_simple_sequence = np.array([])
        self.has_not_usable_index = False

    # parameter I/O
    # this function give trajectory index to this trajectory
    def give_index(self, index1):
        self.trajectory_index = index1

    # this function get index of trajectory
    def get_index(self):
        return self.trajectory_index

    # this function give trajectory point number
    def give_point_number(self, point_number1):
        self.point_number = point_number1

    # this function get point number of trajectory
    def get_point_number(self):
        if self.point_number == 0:
            self.point_number = self.trajectory_array.shape[0]
            if self.point_number == 0:
                raise ValueError('Point number of trajectory {} is zero!'.format(self.trajectory_index))
        return self.point_number

    # this function give trajectory point array by ndarray of trajectory
    def give_trajectory_list(self, point_array):
        self.trajectory_array = point_array

    # this function gets point array form this trajectory
    def get_trajectory_list(self):
        return self.trajectory_array

    #
    def give_level1_index_array(self, level1_array):
        self.level1_cell_index_sequence = level1_array

    #
    def get_level1_index_array(self):
        return self.level1_cell_index_sequence

    #
    def give_single_trajectory_cell_density(self, whole_cell_number):
        level1_index_array = self.get_level1_index_array()
        general_tool1 = GeneralTools()
        # indices, frequency = np.unique(level1_index_array, return_counts=True)
        # wrong_cell_index = indices >= whole_cell_number
        # if wrong_cell_index.any():
        #     raise IndexError('trajectory {} has wrong cell index'.format(self.get_index()))
        # whole_cell_frequency = general_tool1.whole_frequency(indices, frequency, whole_cell_number)
        try:
            whole_cell_frequency = general_tool1.density_of_single_array(whole_cell_number, level1_index_array)
        except IndexError:
            raise IndexError('trajectory {} has wrong cell index'.format(self.get_index()))
        return whole_cell_frequency

    #
    def give_regularized_trajectory_cell_density(self, whole_cell_number):
        raw_frequency = self.give_single_trajectory_cell_density(whole_cell_number)
        point_number = self.get_point_number()
        regularized_frequency = raw_frequency / point_number
        return regularized_frequency

    # this function gives this trajectory a simple discrete trajectory
    def give_simple_trajectory(self, dict1: np.ndarray):
        level2_cell_index_array = self.level2_cell_index_sequence
        unrepeated_sequence, frequency = self.calculate_unrepeated_trajectory(level2_cell_index_array)
        unrepeated_usable_sequence = dict1[unrepeated_sequence]
        self.cell_sequence = unrepeated_sequence
        self.cell_sequence_frequency = frequency
        self.usable_simple_sequence = unrepeated_usable_sequence

    #
    def calculate_unrepeated_trajectory(self, sequence: np.ndarray):
        # index_array = []
        # frequency_array = []
        # index_array.append(sequence[0])
        # repeat_counter = 0
        # for point1 in sequence:
        #     if point1 == index_array[-1]:
        #         repeat_counter += 1
        #     else:
        #         index_array.append(point1)
        #         frequency_array.append(repeat_counter)
        #         repeat_counter = 1
        # frequency_array.append(repeat_counter)
        # index_array = np.asarray(index_array, dtype=int)
        # frequency_array = np.asarray(frequency_array, dtype=int)

        gt1 = GeneralTools()
        index_array, frequency_array = gt1.unreapted_int_array(sequence)
        return index_array, frequency_array

    #
    def give_single_trajectory_subcell_density(self, whole_cell_number: int):
        index_array = self.level2_cell_index_sequence
        general_tool1 = GeneralTools()
        try:
            whole_cell_frequency = general_tool1.density_of_single_array(whole_cell_number, index_array)
            whole_cell_frequency = whole_cell_frequency / self.level1_cell_index_sequence.size
        except IndexError:
            raise IndexError('trajectory {} has wrong subcell index'.format(self.get_index()))
        return whole_cell_frequency




