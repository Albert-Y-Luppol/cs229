import numpy as np

class StohasticGradientDescent:
    def __init__(self, learningRateAlpha = 0.01, alphaDegradationKoeficient = 0.99):
        self.learningRateAlpha = learningRateAlpha
        self.alphaDegradationKoeficient = alphaDegradationKoeficient

    def countWeightParams(self, x, y, options = {}):
        amountOfTrainigSetsM = x.shape[0]
        repeatLearningNOfTimes = 1
        if(options.get('repeatLearningNofTimes')):
            repeatLearningNOfTimes = options.get('repeatLearningNofTimes')
        x0 = np.ones((amountOfTrainigSetsM, 1))
        x = np.hstack((x0, x))
        self.weightTheta = np.zeros((x.shape[1], 1))
        self.costJi = []
        self.__countCost(x, y)
        learningRateAlpha = self.learningRateAlpha
        for _ in range(repeatLearningNOfTimes):
            for i, feature in enumerate(x):
                # print(np.array([feature]))
                hypothese = np.dot(np.array([feature]), self.weightTheta)
                residial = hypothese - y[i]
                self.weightTheta = self.weightTheta - (learningRateAlpha * np.dot(np.array([feature]).T, residial))
                self.__countCost(x, y)
            learningRateAlpha *= self.alphaDegradationKoeficient
        # print(learningRateAlpha)
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
        cost = np.sum(residuals ** 2) / 2
        self.costJi.append(cost)

    def predict(self, x):
        amountOfTrainingSetsM = x.shape[0]
        x0 = np.ones((amountOfTrainingSetsM, 1))
        x = np.hstack((x0, x))
        return np.dot(x, self.weightTheta)