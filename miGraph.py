"""
Author: starplus酱
Home page: https://blog.csdn.net/qq_39228514?spm=1001.2014.3001.5509
Create: 2021 0426
Last modify: 2021 0426
"""


import numpy as np
from Prototype import MIL


class miGraph(MIL):
    def __init__(self, dataset_path, para_gamma=0.1, k=10):
        super(miGraph, self).__init__(dataset_path)
        self.gamma = para_gamma
        self.k = k
        self.affinity_matrics = self.get_affinity_matrics(self.gamma)

    def get_mapping(self):
        sim_matrix = np.zeros((self.num_bags, self.num_bags))
        for i in range(self.num_bags):
            for j in range(self.num_bags):
                sim_matrix[i, j] = self.sim_between_bags(i, j, self.gamma)

        train_idx_dict, test_idx_dict = self.get_index(para_k=self.k)
        for m in range(self.k):
            train_idx = train_idx_dict[m]
            test_idx = test_idx_dict[m]
            train_sim_matrix = sim_matrix[train_idx]
            train_sim_matrix = train_sim_matrix[:, train_idx]
            test_sim_matrix = sim_matrix[test_idx]
            test_sim_matrix = test_sim_matrix[:, train_idx]
            yield train_sim_matrix, self.bags_label[train_idx], test_sim_matrix, self.bags_label[test_idx], None

    def get_affinity_matrics(self, gamma):
        temp_total_matrix = []
        for i in range(self.num_bags):
            temp_total_matrix.append(self.bag2matrix(i, gamma))
        return temp_total_matrix

    def bag2matrix(self, i, gamma):
        ins = self.bags[i, 0][:, :-1]
        num_ins = len(ins)
        dis_matrix = np.zeros((num_ins, num_ins))
        sum_dis = 0
        for j in range(num_ins):
            for k in range(num_ins):
                dis_matrix[j, k] = self.Gaussian_RBF(ins[j], ins[k], gamma=gamma)
                sum_dis += dis_matrix[j, k]
        delta = sum_dis / (num_ins ** 2)
        for j in range(num_ins):
            for k in range(num_ins):
                if dis_matrix[j, k] < delta or j == k:
                    dis_matrix[j, k] = 1
                else:
                    dis_matrix[j, k] = 0
        return dis_matrix

    def sim_between_bags(self, i, j, gamma):
        a_matrix_i = self.affinity_matrics[i]
        number_row_i = self.bags_size[i]
        b_matrix_j = self.affinity_matrics[j]
        num_row_j = self.bags_size[j]
        numerator = 0
        for a in range(number_row_i):
            for b in range(num_row_j):
                numerator += (1 / (np.sum(a_matrix_i[a]) * np.sum(b_matrix_j[b]))) * \
                             self.Gaussian_RBF(self.bags[i, 0][:, :-1][a], self.bags[j, 0][:, :-1][b], gamma=gamma)
        denominator_1 = 0
        for a in range(number_row_i):
            denominator_1 += 1 / np.sum(a_matrix_i[a])
        denominator_2 = 0
        for b in range(num_row_j):
            denominator_2 += 1 / np.sum(b_matrix_j[b])
        denominator = denominator_1 + denominator_2
        return numerator / denominator

    def Gaussian_RBF(self, ins1, ins2, gamma):
        return np.exp(-gamma * np.sum(np.power((ins1 - ins2), 2)))
