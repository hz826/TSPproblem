import numpy as np
import matplotlib.pyplot as plt
from TSPproblem import TSPproblem
plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["font.sans-serif"] = ['SimHei']

TSP = TSPproblem()
# TSP.generate(100)
# TSP.save('g100.txt')
TSP.load('g100.txt')


path, graph = TSP.ACO(100, 200)
TSP.show(131, path, 133, graph, "改进前")

path, graph = TSP.ACO(100, 200, gamma=0.8)
TSP.show(132, path, 133, graph, "改进后")


plt.subplot(133)
plt.xlabel('时间/s')
plt.ylabel('最优解')

plt.legend()
plt.show()