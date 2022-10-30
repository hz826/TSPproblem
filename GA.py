import numpy as np
import random
import time

def GA(self) :
    MAX_iter = 400
    individual_num = 60
    mutate_prob = 0.25

    def cross(individual_list) :
        new_gen = []
        np.random.shuffle(individual_list)

        for i in range(0, len(individual_list)-1, 2) :
            g0 = individual_list[i].copy()
            g1 = individual_list[i+1].copy()

            index = np.sort(np.random.permutation(range(1,self.n))[:2])

            pos0_recorder = {value: idx for idx, value in enumerate(g0)}
            pos1_recorder = {value: idx for idx, value in enumerate(g1)}

            # 交叉
            for j in range(index[0], index[1]) :
                value0, value1 = g0[j], g1[j]
                pos0, pos1 = pos0_recorder[value1], pos1_recorder[value0]

                g0[j], g0[pos0] = g0[pos0], g0[j]
                g1[j], g1[pos1] = g1[pos1], g1[j]

                pos0_recorder[value0], pos0_recorder[value1] = pos0, j
                pos1_recorder[value0], pos1_recorder[value1] = j, pos1
            
            new_gen.append(g0)
            new_gen.append(g1)
        
        return new_gen

    def mutate(individual_list) :
        new_gen = []
        for individual in individual_list :
            if np.random.random() < mutate_prob :
                # 翻转切片
                old_genes = individual.copy()
                index = np.sort(np.random.permutation(range(1,self.n))[:2])

                individual = np.concatenate((old_genes[:index[0]], np.flip(old_genes[index[0]:index[1]]), old_genes[index[1]:]))
            # 两代合并
            new_gen.append(individual)
        
        return new_gen

    def select(individual_list) :
        # 锦标赛
        group_num = 10  # 小组数
        group_size = 10  # 每小组人数
        group_winner = individual_num // group_num  # 每小组获胜人数
        winners = []  # 锦标赛结果
        
        for i in range(group_num):
            group = []
            for j in range(group_size):
                # 随机组成小组
                player = random.choice(individual_list)
                group.append(player)
            
            group = sorted(group, key = lambda individual : self.TSPdistance(individual))
            # 取出获胜者
            winners += group[:group_winner]
        return winners

    ## GA main
    # 初代种群
    start_time = time.time()
    times = []
    scores = []

    best_path = np.random.permutation(range(self.n))  # 每一代的最佳个体
    individual_list = [np.random.permutation(range(self.n)) for _ in range(individual_num)]

    best_path = individual_list[0]
    # 迭代
    for iter in range(MAX_iter) :
        print('iter :', iter)

        # 交叉
        new_gen = cross(individual_list)
        # 变异
        new_gen = mutate(new_gen)
        # 选择
        new_gen = new_gen + individual_list
        individual_list = select(new_gen)
        # 获得这一代的结果

        best_path = individual_list[np.argmax([self.TSPdistance(individual) for individual in individual_list])].copy()
        
        if len(times) == 0 or time.time()-start_time - times[-1] > 0.1 :
            times.append(time.time() - start_time)
            scores.append(self.TSPdistance(best_path))

    return best_path, (times, scores)