import numpy as np

from helpers.plotter import Plotter
from helpers.traningSetGenerator import TrainingSetsGenerator


class BatchGradientDescent:
  def __init__(self, learningRateAlpha = 0.01):
    self.learningRateAlpha = learningRateAlpha

  def countWeightParams(self, x, y):
    amountOfTrainingSetsM = x.shape[0]
    x0 = np.ones((amountOfTrainingSetsM, 1))
    x = np.hstack((x0, x))
    # print(x)
    self.weightTheta = np.zeros((x.shape[1], 1))
    # print(self.weightTheta)
    self.costJi = []
    self.__countCost(x, y)
    for _ in range(amountOfTrainingSetsM):
      hypotheses = np.dot(x, self.weightTheta)
      # print(hypotheses)
      # print(np.dot((hypotheses - y).T, x).T) # x*r => xT (dot) r
      residuals = hypotheses - y
      # print(residuals)
      # print(np.dot(x.T, residuals) / amountOfTrainingSetsM)
      # print(np.dot(x.T, residuals))
      # print(self.weightTheta)
      self.weightTheta = self.weightTheta -  (self.learningRateAlpha)*np.dot(x.T, residuals) / amountOfTrainingSetsM
      # print(self.weightTheta)
      self.__countCost(x,y)

  def __countCost(self, x, y):
    residuals = np.dot(x, self.weightTheta) - y
    # print(residuals)
    cost = np.sum(residuals ** 2) / 2
    # print(cost)
    self.costJi.append(cost)

  def predict(self, x):
    amountOfTrainingSetsM = x.shape[0]
    x0 = np.ones((amountOfTrainingSetsM, 1))
    x = np.hstack((x0, x))
    return np.dot(x, self.weightTheta)

# a = [[1, 2, 3], [3, 4, 5]]
# b = [[1], [2], [3]]
# print(a, ' + ', b, ' = ', np.dot(a, b))

# a = np.random.rand(2, 3)
# b = [[1], [2], [3]]
# print(a, ' + ', b, ' = ', np.dot(a, b))

# a = np.random.rand(2, 3)
# b = np.random.rand(3, 1)
# print(a, ' + ', b, ' = ', np.dot(a, b))

bgd = BatchGradientDescent()
tsg = TrainingSetsGenerator()
# p = Plotter()
[x, y] = tsg.generateTrainingSets()
# print(x,y)
bgd.countWeightParams(x,y)
print('Result Theta:')
print(bgd.weightTheta)
print('Costs Initial:')
print(bgd.costJi[1])
print('Cost Final:')
print(bgd.costJi[-1])


# predictedY = bgd.predict(x)

# p.addInitialDots(x,y)
# p.addLine(x,predictedY)
# p.show()
