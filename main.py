from dots_drawer import DotsDrawer
from perceptron import Perceptron
from random import uniform

test_data = [
    # zeroes
    ((15, 15), 0),
    ((45, 3), 0),
    ((78, 100), 0),
    ((249, 249), 0),
    ((1, 1), 0),
    ((52, 25), 0),
    ((94, 96), 0),
    ((224, 111), 0),
    # ones
    ((260, 260), 1),
    ((451, 278), 1),
    ((345, 453), 1),
    ((301, 270), 1),
    ((488, 488), 1),
    ((400, 263), 1),
    ((370, 390), 1),
    ((490, 450), 1),
]


def get_random_dot():
    return int(uniform(0, 500)), int(uniform(0, 500))


if __name__ == '__main__':
    colours = {0: (0, 240, 0),
               1: (0, 0, 240)}
    with DotsDrawer('dots.png') as drawer:
        for dot, color in test_data:
            drawer.dot(x=dot[0], y=dot[1], color=colours[color])

        perc = Perceptron()
        for _ in range(10):
            for dot, color in test_data:
                res = perc.eval(list(dot))
                err = 0.5 * (color - res) ** 2
                perc.teach(err=err, data=list(dot))

        for _ in range(100):
            rand_dot = get_random_dot()
            if perc.eval(list(rand_dot)) > 0.5:
                drawer.dot(x=rand_dot[0], y=rand_dot[1], color=colours[1])
            else:
                drawer.dot(x=rand_dot[0], y=rand_dot[1], color=colours[0])
