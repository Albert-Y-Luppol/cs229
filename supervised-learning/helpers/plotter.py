import matplotlib.pyplot as plt


class Plotter:
  def __init__(self):
    plt.xlabel('x')
    plt.ylabel('y')

  def addInitialDots(self, x, y, scale=1):
    plt.scatter(x, y, scale)

  def addLine(self, x, y):
    plt.plot(x, y)

  def show(self):
    plt.show()
