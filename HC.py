import numpy as np
import itertools
import time

def HC(self, MAX_iter) : # 爬山算法
    start_time = time.time()
    times = []
    scores = []

    best_path = np.random.permutation(range(self.n))

    for iter in range(MAX_iter) :
        if iter % 10000 == 0 :
            print('iter :', iter)
        
        next_path = best_path.copy()

        index = np.sort(np.random.permutation(range(1,self.n))[:3])
        p = [next_path[index[2]:        ],
             next_path[        :index[0]],
             next_path[index[0]:index[1]],
             next_path[index[1]:index[2]]]
        
        (best_q, best_length) = ([], 1e9)
        for q in itertools.permutations([1,2,3]) :
            qq = [0] + list(q)
            length_now = sum(self.G[p[qq[i-1]][-1]][p[qq[i]][0]] for i in range(4))

            if length_now < best_length :
                (best_q, best_length) = (qq, length_now)
        
        next_path = np.concatenate([p[best_q[i]] for i in range(4)])

        if self.TSPdistance(next_path) < self.TSPdistance(best_path) :
            best_path = next_path.copy()
        
        if len(times) == 0 or time.time()-start_time - times[-1] > 0.1 :
            times.append(time.time() - start_time)
            scores.append(self.TSPdistance(best_path))
    
    print('HC :', self.TSPdistance(best_path))
    return (best_path, (times, scores))