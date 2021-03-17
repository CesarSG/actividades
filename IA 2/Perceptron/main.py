import random
import numpy as np
import Perceptron
import matplotlib.pyplot as plt

x_input = []
d_input = []


def getPoints():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

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

    heta = 0.4
    max_epoch = 500

    getPoints()

    x = np.vstack((x_input))
    d = np.vstack((d_input))
    x_shape = x.shape
    w = np.random.rand(x_shape[1])

    print("--------------Valores ingresados--------------")
    print(x)
    print(d)
    print(w)
    print("----------------------------------------------")
    input()

    practica = Perceptron.Perceptron(x, w, d, heta, max_epoch)
    practica.run()


if __name__ == '__main__':
    main()
