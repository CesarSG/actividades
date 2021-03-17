import random
import numpy as np
import NeuralNetwork
import Helpers
import matplotlib.pyplot as plt

x_input = []
d_input = []


def getPoints():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])

    def onclick(event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))

        if event.button == 1:
            plt.plot(event.xdata, event.ydata, 'bo')
            x_input.append([1, event.xdata, event.ydata])
            d_input.append([1])

        if event.button == 3:
            plt.plot(event.xdata, event.ydata, 'ro')
            x_input.append([1, event.xdata, event.ydata])
            d_input.append([0])

        if event.button == 2:
            print("x: ", x_input)
            print("d: ", d_input)

        fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


def main():

    heta = 0.1
    max_epoch = 25000
    min_error = 0.001
    neurons = 8
    showEvery = 300

    getPoints()

    x = np.vstack((x_input))
    d = np.vstack((d_input))

    practice = NeuralNetwork.NeuralNetwork(
        x, d, heta, max_epoch, min_error, neurons, showEvery)
    practice.run()


if __name__ == '__main__':
    main()
