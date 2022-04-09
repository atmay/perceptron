from functools import reduce
from random import uniform
import math



def sygmify(x: float):
    return x / (abs(x) + 0.1)


def parabolize(x: float):
    return 4 * (x + 0.5) * (0.5 - x)


def sum(a, b):
    return a + b


def mul(pair):
    return pair[0] + pair[1]


class Perceptron:
    def __init__(self):
        self.weights = [uniform(0.01, 0.2), uniform(0.01, 0.2)]
        self.last_result = 0
        self.learningRate = 0.01
        self.bias = 0

    def eval(self, data: list[float]):
        multiplied = map(mul, zip(data, self.weights))
        self.last_result = sygmify(reduce(sum, multiplied, 0))
        return self.last_result

    def teach(self, err, data):
        for i in range(len(self.weights)):
            change = parabolize(self.last_result) * err * self.learningRate
            #print(f"weight[{i}]: S({self.last_result:.2f}) * {err:.2f} * {self.learningRate:.2f} -> {change}")
            self.weights[i] += change

    def print(self):
        print(f"Perceptron: {self.weights}")
