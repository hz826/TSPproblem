import numpy as np
import matplotlib.pyplot as plt

class TSPproblem :
    def __init__(self) :
        pass

    def get_distance(self) :
        self.G = np.zeros((self.n, self.n))
        for i in range(self.n) :
            for j in range(self.n) :
                self.G[i][j] = np.linalg.norm(self.points[i] - self.points[j])

    def generate(self, n) :
        self.n = n
        self.points = np.random.rand(n, 2)
        self.get_distance()
    
    def save(self, filename) :
        np.savetxt(filename, self.points)
    
    def load(self, filename) :
        self.points = np.loadtxt(filename)
        self.n = self.points.shape[0]
        assert self.points.shape == (self.n, 2)
        self.get_distance()

    def show(self, subplot_id, path=[]) :
        X = self.points.T[0]
        Y = self.points.T[1]

        plt.subplot(subplot_id)
        plt.scatter(X, Y)

        if path.any() :
            # print(path)
            XX = [X[path[i]] for i in range(-1,self.n)]
            YY = [Y[path[i]] for i in range(-1,self.n)]
            plt.plot(XX, YY)
    
    def TSPdistance(self, path) :
        return sum(self.G[path[i-1]][path[i]] for i in range(self.n))
    
    from HC import HC
    from ACO import ACO