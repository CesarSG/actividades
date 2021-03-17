import os
import sys
import csv
import Helpers
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, x, d, heta, max_epoch, min_error, neurons, showEvery):
        self._x = x
        self._d = d
        self._w_hidden = np.random.rand(self._x.shape[1], neurons)
        self._w_output = np.random.rand(neurons, 1)
        self.output = np.zeros(d.shape)
        self._heta = heta
        self._max_epoch = max_epoch
        self._min_error = min_error
        self._showEvery = showEvery

    def run(self):
        self.neuralNetwork()

    def outputs(self, output):

        with open('outputs-nn.csv', 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            for res in output:
                filewriter.writerow(res)

    def forwardGraph(self, patterns):
        self.layer_hidden = self.activation(np.dot(patterns, self._w_hidden))
        self.layer_output = self.activation(
            np.dot(self.layer_hidden, self._w_output))
        return self.layer_output

    def plotContour(self):

        feature_x = np.linspace(-5.0, 5.0, 41)
        feature_y = np.linspace(-5.0, 5.0, 41)

        [X, Y] = np.meshgrid(feature_x, feature_y)

        info = []
        for x, y in zip(X, Y):
            for i in range(41):
                info.append([1, x[i], y[i]])
        info = np.vstack((info))

        result = self.forwardGraph(info)

        j = 0
        result_nn = []
        for i in range(len(result)):
            if j == 40:
                result_nn.append(
                    [result.item(i-40), result.item(i-39), result.item(i-38), result.item(i-37), result.item(i-36), result.item(i-35), result.item(i-34), result.item(i-33), result.item(i-32), result.item(i-31), result.item(i-30), result.item(i-29), result.item(i-28), result.item(i-27), result.item(i-26), result.item(i-25), result.item(i-24), result.item(i-23), result.item(i-22), result.item(i-21), result.item(i-20), result.item(i-19), result.item(i-18), result.item(i-17), result.item(i-16), result.item(i-15), result.item(i-14), result.item(i-13), result.item(i-12), result.item(i-11), result.item(i-10), result.item(i-9), result.item(i-8), result.item(i-7), result.item(i-6), result.item(i-5), result.item(i-4), result.item(i-3), result.item(i-2), result.item(i-1), result.item(i)])
                j = 0
            else:
                j += 1

        result_nn = np.vstack((result_nn))

        fig, ax = plt.subplots(1, 1)

        # plots filled contour plot
        ax.contourf(X, Y, result_nn)

        ax.set_title('Red Multicapa')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        for i, d in zip(self._x, self._d):
            if d > 0.5:
                plt.plot(i[1], i[2], 'bo')
            else:
                plt.plot(i[1],  i[2], 'ro')

        plt.draw()
        plt.pause(1)

    def plotErrors(self, errors):

        epoch = 0
        for error in errors:
            e = epoch
            x = e
            y = error

            plt.plot(x, y, 'ro-')
            plt.title("Errores Multicapa")
            plt.xlabel("Epocas")
            plt.ylabel("Promedio")
            epoch += 1
        plt.show()

    def activation_derivative(self, res):
        return res * (1 - res)

    def activation(self, res):
        return 1/(1 + np.exp(-res))

    def forward(self):
        self.layer_hidden = self.activation(np.dot(self._x, self._w_hidden))
        self.layer_output = self.activation(
            np.dot(self.layer_hidden, self._w_output))
        return self.layer_output

    def backward(self):

        errors = self._d - self.output

        gradient = errors * self.activation_derivative(self.output)
        d_output = self._heta * gradient * self.layer_output

        gradient_hidden = np.dot(d_output, self._w_output.T) * \
            self.activation_derivative(self.layer_hidden)

        d_hidden = self._heta * gradient_hidden * self.layer_hidden

        self._w_output += np.dot(self.layer_hidden.T, gradient)
        self._w_hidden += np.dot(self._x.T, gradient_hidden)

    def train(self):
        self.output = self.forward()
        self.backward()

    def neuralNetwork(self):

        errors = []

        for epoch in range(self._max_epoch):
            if epoch % self._showEvery == 0:
                error = np.mean(np.square(self._d - self.forward()))
                print("\n")
                print("--------------- Epoca: " +
                      str(epoch) + " -----------------------")
                #print("Salida: \n" + str(self.forward()))
                print("Error: ", error)
                errors.append(error)
                self.plotContour()
                plt.close()
            if (np.mean(np.square(self._d - self.forward())) < self._min_error):
                break

            self.train()

        self.outputs(self.forward())
        print(epoch)
        self.plotContour()
        input()
        plt.close('all')
        self.plotErrors(errors)
        input()
