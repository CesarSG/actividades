import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, x, w, d, heta, max_epoch):
        self._x = x
        self._w = w
        self._d = d
        self._heta = heta
        self._max_epoch = max_epoch
        self._trained = False

    def run(self):
        self.perceptron()

    def outputs(self):
        result = np.dot(self._x, self._w)
        outputs = []

        with open('outputs-perceptron.csv', 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            for res in result:
                output = self.step(res)
                outputs.append(output)
                filewriter.writerow([output])

        return outputs

    def prodError(self, errors):
        error = (errors**2)
        return error

    def plotError(self, historial):
        epoch = list(range(len(historial)))
        plt.plot(epoch, historial)
        plt.show()

    def plot(self, w, x, des):

        m = (-w[1]/w[2])
        b = -(w[0]/w[2])

        y_0 = (m*0)+(b)
        y_1 = (m*1)+(b)

        x1, y1 = [0, 1], [y_0, y_1]
        plt.plot(x1, y1)

        for i, d in zip(x, des):
            if d > 0:
                plt.plot(i[1], i[2], 'bo')
            else:
                plt.plot(i[1],  i[2], 'ro')

        plt.draw()
        plt.pause(0.5)
        plt.clf()

    def step(self, y):
        return 1/(1 + np.exp(-y))

    def train(self, result):
        i = 0
        errors = 0

        for pattern, res in zip(self._x, result):
            y = self.step(res)
            error = self._d[i] - y
            i += 1

            if(error != 0):
                self._w += self._heta * error * pattern * y * (1-y)
                errors += 1

        return errors

    def perceptron(self):

        historial = []

        for epoch in range(self._max_epoch):

            print("Número de epoca: ", epoch)
            result = np.dot(self._x, self._w)
            print("\t", result)
            errors = self.train(result)
            prod = self.prodError(errors)
            historial.append(prod)
            self.plot(self._w, self._x, self._d)

            print("\t Errores: ", errors)
            print("------------------")
            if(errors == 0):
                self._trained = True
            if(self._trained):
                input()
                break

        plt.close()
        self.plotError(historial)

        outputs = self.outputs()
        print("Outputs: ", outputs)
