import numpy as np

class TrainingSetsGenerator:
  def __init__(self, amountOfTriningExamplesM = 10000, amountOfFeaturesX = 3, frontier = 10, accuracy = 0):
    self.m = amountOfTriningExamplesM
    self.n = amountOfFeaturesX +  1
    self.max = frontier
    self.delta = accuracy

  def generateTrainingSets(self):
    x = np.random.rand(self.m, self.n) * self.max
    x[:, 0] = 1
    delta = np.random.rand(self.m, 1) * (self.delta * 2) - self.delta
    initialWeight = np.random.rand(self.n, 1) * self.max - self.max / 2
    print('Initial Theta:')
    print(initialWeight)
    yClear = np.dot(x, initialWeight)
    y = yClear - delta
    x = x[:, 1:]
    # print('X:')
    # print(x)
    # print('Y:')
    # print(y)
    return [x, y]