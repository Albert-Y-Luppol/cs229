import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionUsingGradientDescent:
    """Linear Regression Using Gradient Descent.
    Parameters
    ----------
    learningParamenter : float
        Learning rate
    numberOfIterations : int
        No of passes over the training set
    Attributes
    ----------
    weight : weights/ after fitting the model
    cost : total error of the model after each iteration
    """

    def __init__(self, learningParamenter=0.1, numberOfIterations=290):
      self.learningParamenter = learningParamenter
      self.numberOfIterations = numberOfIterations

    def fit(self, x, y):
      """Fit the training data
      Parameters
      ----------
      x : array-like, shape = [n_samples, n_features]
          Training samples
      y : array-like, shape = [n_samples, n_target_values]
          Target values
      Returns
      -------
      self : object
      """

      self.cost = []
      self.weight = np.zeros((x.shape[1], 1))
      m = x.shape[0]

      for _ in range(self.numberOfIterations):
        yPred = np.dot(x, self.weight)
        residuals = yPred - y
        gradientVector = np.dot(x.T, residuals)
        self.weight -= (self.learningParamenter / m) * gradientVector
        cost = np.sum((residuals ** 2) / (2 * m))
        self.cost.append(cost)
      return self

    def predict(self, x):
      """ Predicts the value after the model has been trained.
      Parameters
      ----------
      x : array-like, shape = [n_samples, n_features]
          Test samples
      Returns
      -------
      Predicted value
      """
      return np.dot(x, self.weight)


# generate random data-set
np.random.seed(0)
x = np.random.rand(100, 1)
y =  25 * x

# model

model = LinearRegressionUsingGradientDescent()
trainedModel = model.fit(x, y)

# plot
plt.scatter(x, y, s=10)
plt.plot(x, trainedModel.predict(x))
plt.xlabel('x')
plt.ylabel('y')
plt.show()
