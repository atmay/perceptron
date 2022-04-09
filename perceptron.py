from functools import reduce
from random import uniform


class Perceptron:
    def __init__(self):
        self.weights = [uniform(0.01, 0.2), uniform(0.01, 0.2)]
        self.res = 0
        self.coeff = 0.1

    @staticmethod
    def sygmify(x: float):
        return x / (abs(x) + 0.1)

    @staticmethod
    def parabolize(x: float):
        res = 5 * (x + 0.5) * (0.5 - x)
        if res < 0:
            return 0
        return res

    def eval(self, data: list[float]):
        multiplied = map(lambda a: a[0] * a[1], zip(data, self.weights))
        self.res = reduce(lambda b, c: b + c, multiplied, 0)
        return self.res

    def teach(self, err, data):
        for i in range(len(self.weights)):
            self.weights[i] -= self.parabolize(data[i]) * err * self.coeff
