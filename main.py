import numpy as np
import matplotlib.pyplot as plt
from TSPproblem import TSPproblem

TSP = TSPproblem()
# TSP.generate(10)
# TSP.save('g10.txt')
TSP.load('g60.txt')


path, graph = TSP.HC(500000)
TSP.show(131, path)
plt.subplot(133)
plt.plot(graph[0], graph[1])


path, graph = TSP.ACO(150, 200)
TSP.show(132, path)
plt.subplot(133)
plt.plot(graph[0], graph[1])

 
plt.show()