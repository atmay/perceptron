from functools import reduce
from random import uniform
import math



def legacy_sigmoid(x: float):
    return x / (abs(x) + 0.1)


def legacy_sigmoid_derivative(x: float):
    return 4 * (x + 0.5) * (0.5 - x)


def sigmoid(x: float):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x: float):
    return x * (1 - x)


class Perceptron:
    def __init__(self, size):
        self.weights = [uniform(0.01, 0.2) for _ in range(size)]
        self.last_result = 0
        self.learningRate = 0.1
        self.bias = 0

    def eval(self, f_input: list[float]):
        self.last_result = 0
        for inp, weight in zip(f_input, self.weights):
            self.last_result += inp * weight
        self.last_result += self.bias
        self.last_result = sigmoid(self.last_result)
        return self.last_result

    def teach(self, f_input: list[float], err: float):
        # print(f"teach( {f_input} , {err} )")
        change = err * sigmoid_derivative(self.last_result) * self.learningRate
        for i in range(len(self.weights)):
            #print(f"weight[{i}]: S({self.last_result:.2f}) * {err:.2f} * {self.learningRate:.2f} -> {change}")
            self.weights[i] += f_input[i] * change
        self.bias += change

    def print(self):
        print(f"Perceptron: [ {self.bias} ] . {self.weights}")
