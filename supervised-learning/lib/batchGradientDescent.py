import numpy as np

class BatchGradientDescent:
  def __init__(self, learningRateAlpha = 0.01):
    self.learningRateAlpha = learningRateAlpha

  def countWeightParams(self, x, y, options = {}):
    #  (printFinalTheta, printCostInitial, printCostFinal) = options
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

    if (options.get('printFinalTheta')):
      print('Result Theta:')
      print(self.weightTheta)
    if (options.get('printCostInitial')):
      print('Costs Initial:')
      print(self.costJi[1])
    if (options.get('printCostFinal')):
      print('Cost Final:')
      print(self.costJi[-1])

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
