import numpy as np

from helpers.plotter import Plotter
from helpers.traningSetGenerator import TrainingSetsGenerator

from lib.batchGradientDescent import BatchGradientDescent

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
p = Plotter()
(x, y) = tsg.generateTrainingSets({ 'printTheta': True })
# print(x,y)
bgd.countWeightParams(x,y, { 'printFinalTheta': True, 'printCostInitial': True, 'printCostFinal': True })

if(x.shape[1] == 1):
  predictedY = bgd.predict(x)
  p.addInitialDots(x,y)
  p.addLine(x, predictedY)
  p.show()
