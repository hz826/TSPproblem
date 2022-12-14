import numpy as np
import time

def ACO(self, AntCount, MAX_iter, alpha=1, beta=2, gamma=1, Q=1, rho=0.1) : # 蚁群算法
    start_time = time.time()
    times = []
    scores = []
    
    # 信息素
    # alpha = 1 # 信息素重要程度因子
    # beta = 2  # 启发函数重要程度因子
    # gamma = 1 # 负反馈因子
    # rho = 0.1 # 挥发速度
    # Q = 1
    # 初始信息素矩阵，全是为1组成的矩阵
    pheromonetable = np.ones((self.n, self.n))

    # 候选集列表,存放100只蚂蚁的路径(一只蚂蚁一个路径),一共就Antcount个路径，一共是蚂蚁数量*城市数量
    next_path = np.zeros((AntCount, self.n), dtype=np.int32)

    best_path = np.random.permutation(range(self.n))
    best_length = self.TSPdistance(best_path)

    # 倒数矩阵
    # etable = 1.0 / self.G

    for iter in range(MAX_iter) :
        if iter % 10 == 0 :
            print('iter :', iter)

        # first：蚂蚁初始点选择
        tmp = np.array([], dtype=np.int32)
        while tmp.size < AntCount :
            tmp = np.append(tmp, np.random.permutation(range(self.n)))
        next_path[:, 0] = tmp[:AntCount]

        length = np.zeros(AntCount) #每次迭代的N个蚂蚁的距离值

        P = np.zeros((self.n, self.n))
        for i in range(self.n) :
            for j in range(self.n) :
                if i != j :
                    # 计算当前城市到剩余城市的（信息素浓度^alpha）*（城市适应度的倒数）^beta
                    # 预处理减少重复运算
                    P[i][j] = np.power(pheromonetable[i][j], alpha) * \
                              np.power(1.0 / self.G[i][j], beta)

        # second：选择下一个城市选择
        for i in range(AntCount) :
            now = next_path[i][0]  # 当前所在点,第i个蚂蚁在第一个城市
            unvisit = [i for i in range(self.n) if i != now]

            for j in range(1, self.n) : # 访问剩下的n个城市，n次访问
                # 计算当前城市到剩余城市的转移概率
                protrans = [P[now][next] for next in unvisit]
                protrans /= sum(protrans)

                # 轮盘赌选择
                next = np.random.choice(unvisit, p=protrans)

                # 下一个访问城市的索引值
                next_path[i, j] = next
                unvisit.remove(next)
                now = next  # 更改出发点，继续选择下一个到达点
            
            length[i] = self.TSPdistance(next_path[i])

            if length[i] < best_length :
                best_path = next_path[i].copy()
                best_length = length[i]

        """
            信息素的更新
        """

        AntID = sorted([i for i in range(AntCount)], key = lambda x : length[x])

        #信息素的增加量矩阵
        change = np.zeros((self.n, self.n))
        for ii in range(AntCount) :
            i = AntID[ii]
            for j in range(self.n) :
                # 当前路径之间的信息素的增量：1/当前蚂蚁行走的总距离的信息素
                change[next_path[i][j-1]][next_path[i][j]] += Q / length[i] * (1 if ii < gamma*AntCount else -1)
        
        #信息素更新的公式：
        pheromonetable = (1 - rho) * pheromonetable + change
        pheromonetable[pheromonetable < 0.00001] = 0.00001

        if len(times) == 0 or time.time()-start_time - times[-1] > 0.1 :
            times.append(time.time() - start_time)
            scores.append(self.TSPdistance(best_path))
    
    print(times)
    print(scores)
    print('ACO :', self.TSPdistance(best_path))
    return (best_path, (times, scores))