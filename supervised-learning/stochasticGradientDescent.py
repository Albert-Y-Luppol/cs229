import numpy as np

from helpers.plotter import Plotter
from helpers.traningSetGenerator import TrainingSetsGenerator

from lib.stohasticGradientDescent import StohasticGradientDescent

# a = [[1, 2, 3], [3, 4, 5]]
# b = [[1], [2], [3]]
# print(a, ' + ', b, ' = ', np.dot(a, b))

# a = np.random.rand(2, 3)
# b = [[1], [2], [3]]
# print(a, ' + ', b, ' = ', np.dot(a, b))

# a = np.random.rand(2, 3)
# b = np.random.rand(3, 1)
# print(a, ' + ', b, ' = ', np.dot(a, b))

sgd = StohasticGradientDescent(0.005, 1)
tsg = TrainingSetsGenerator(100, 1, 10, 0)
p = Plotter()
(x, y) = tsg.generateTrainingSets({ 'printTheta': True })
# print(x,y)
sgd.countWeightParams(x,y, {
    'printFinalTheta': True,
    'printCostInitial': True,
    'printCostFinal': True,
    'repeatLearningNofTimes': 100,
    })

if(x.shape[1] == 1):
  predictedY = sgd.predict(x)
  p.addInitialDots(x,y)
  p.addLine(x, predictedY)
  p.show()
