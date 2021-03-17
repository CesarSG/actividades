import math
import numpy as np
import matplotlib.pyplot as plt


class Activacion:
    def __init__(self):
        self._escalon = [1, 'Función escalon']
        self._lineal_tramos = [2, 'Función lineal a tramos']
        self._lineal = [3, 'Función lineal']
        self._sigmoidal = [4, 'Función Sigmoidal']
        self._tanh = [5, 'Función Tangente Hiperbolica']
        self._gaussiana = [6, 'Función Gaussiana']
        self._sinusoidal = [7, 'Función Sinusoidal']

        self._is_escalon = 1
        self._is_lineal_tramos = 2
        self._is_lineal = 3
        self._is_sigmoidal = 4
        self._is_tanh = 5
        self._is_gaussiana = 6
        self._is_sinusoidal = 7

    def run(self):
        self.plot(self._escalon)
        self.plot(self._lineal)
        self.plot(self._lineal_tramos)
        self.plot(self._sigmoidal)
        self.plot(self._gaussiana)
        self.plot(self._tanh)
        self.plot(self._sinusoidal)

    def escalon(self, x):
        if(x >= 0):
            return 1
        else:
            return 0

    def lineal(self, x):
        return x

    def escalon_tramos(self, x):
        if (x >= 1):
            return 1
        elif (x > 0 and x < 1):
            return x
        else:
            return 0

    def gaussiano(self, x, alpha, r):
        return 1./(math.sqrt(alpha**math.pi))*np.exp(-alpha*np.power((x - r), 2.))

    def plot(self, funcion):

        if(self._is_escalon == funcion[0]):
            x = np.arange(-5.0, 5.0, 0.01)
            plt.plot(x, [self.escalon(i) for i in x])
            plt.xlim(-4, 4)
            plt.ylim(-4, 4)

        if(self._is_lineal == funcion[0]):
            x = np.arange(-5.0, 5.0, 0.1)
            plt.plot(x, [self.lineal(i) for i in x])
            plt.xlim(-4, 4)
            plt.ylim(-4, 4)

        if(self._is_lineal_tramos == funcion[0]):
            x = np.arange(-5.0, 5.0, 0.1)
            plt.plot(x, [self.escalon_tramos(i) for i in x])
            plt.xlim(-4, 4)
            plt.ylim(-4, 4)

        if(self._is_sigmoidal == funcion[0]):
            x = np.linspace(-10, 10, 100)
            z = 1/(1 + np.exp(-x))
            plt.plot(x, z)

        if(self._is_tanh == funcion[0]):
            x = np.linspace(-10, 10, 100)
            z = np.tanh(x)
            plt.plot(x, z)

        if(self._is_gaussiana == funcion[0]):
            x = np.linspace(-3, 3, 100)
            plt.plot(x, self.gaussiano(x, 1, 0))

        if(self._is_sinusoidal == funcion[0]):
            x = np.arange(0, 10, 0.1)
            z = np.sin(x)
            plt.plot(x, z)

        plt.axhline(0, color="black")
        plt.axvline(0, color="black")
        plt.title(funcion[1])
        plt.show()
