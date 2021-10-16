import numpy as np

class TrainingSetsGenerator:
  def __init__(self, amountOfTriningExamplesM = 10000, amountOfFeaturesX = 3, frontier = 10, accuracy = 0):
    self.m = amountOfTriningExamplesM
    self.n = amountOfFeaturesX +  1
    self.max = frontier
    self.delta = accuracy

  def generateTrainingSets(self, options = {}):
    # print(options)
    x = np.random.rand(self.m, self.n) * self.max
    x[:, 0] = 1
    delta = np.random.rand(self.m, 1) * (self.delta * 2) - self.delta
    initialWeight = np.random.rand(self.n, 1) * self.max - self.max / 2
    if (options.get('printTheta')):
      print('Initial Theta:')
      print(initialWeight)
    yClear = np.dot(x, initialWeight)
    y = yClear - delta
    x = x[:, 1:]
    if (options.get('printX')):
      print('X:')
      print(x)
    if (options.get('printY')):
      print('Y:')
      print(y)
    return (x, y)