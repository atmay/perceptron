from random import uniform
import math


class Perceptron:
    def __init__(self, size):
        self.weights = [uniform(0.01, 0.2) for _ in range(size)]
        self.last_result = 0
        self.learning_rate = 0.1
        self.bias = 0

    @staticmethod
    def sigmoid(x: float):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x: float):
        return x * (1 - x)

    def eval(self, f_input: list[float]):
        self.last_result = 0
        for inp, weight in zip(f_input, self.weights):
            self.last_result += inp * weight
        self.last_result += self.bias
        self.last_result = self.sigmoid(self.last_result)
        return self.last_result

    def learn(self, data: list[float], err: float):
        change = err * self.sigmoid_derivative(self.last_result) * self.learning_rate
        for i in range(len(self.weights)):
            self.weights[i] += data[i] * change
        self.bias += change
        res = [w * err for w in self.weights]
        return res

    def print(self):
        print(f"Perceptron: [ {self.bias} ] . {self.weights}")
