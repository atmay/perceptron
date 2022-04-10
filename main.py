from dots_drawer import DotsDrawer
from perceptron import *
from random import uniform

# dots are green and blue (could be changed by changing this constant)
COLOURS = {
    0: (0, 240, 0),
    1: (0, 0, 240)
}

# data for perceptron's learn process. Could be automatically generated.
INITIAL_DATA = [
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


def normalize_dot(dot):
    return dot[0] / 500.0, dot[1] / 500.0


def main_2d(drawer):
    for dot, color in INITIAL_DATA:
        drawer.dot(x=dot[0], y=dot[1], color=COLOURS[color])

    perc = Perceptron(2)
    for i in range(1_000_000):
        max_error = 0
        for dot, answer in INITIAL_DATA:
            res = perc.eval(list(normalize_dot(dot)))
            err = answer - res
            max_error = max(max_error, err)
            perc.learn(err=err, data=list(normalize_dot(dot)))
        perc.learning_rate = max_error * 0.001

    for _ in range(100):
        rand_dot = get_random_dot()
        if perc.eval(list(normalize_dot(rand_dot))) > 0.5:
            drawer.dot(x=rand_dot[0], y=rand_dot[1], color=COLOURS[1], size=2)
        else:
            drawer.dot(x=rand_dot[0], y=rand_dot[1], color=COLOURS[0], size=2)


def main_1d(drawer):
    training_data = []
    for _ in range(20):
        value = uniform(0, 1)
        answer = 0 if value < 0.5 else 1
        entry = ([value], answer)
        training_data.append(entry)

    perc = Perceptron(1)
    for i in range(10_000):
        max_error = 0
        for dot, answer in training_data:
            result = perc.eval(list(dot))
            error = answer - result
            perc.learn(data=list(dot), err=error)
            max_error = max(max_error, error)
        perc.learning_rate = max_error * 0.1

    for dot, answer in training_data:
        drawer.dot(x=50 + dot[0] * 400, y=350, color=COLOURS[answer])

    for _ in range(100):
        dot = [uniform(0, 1)]
        answer = 0 if perc.eval(dot) < 0.5 else 1
        drawer.dot(x=50 + dot[0] * 400, y=400, color=COLOURS[answer])


if __name__ == '__main__':
    """
    Select Perceptron option (1d or 2d) and get it's result saved into a file 'dots.png'
    Currently there are 2 options - 1d which means dots on a single axis and 2d which means dots 
    spread out down x and y axes.
    """
    with DotsDrawer('dots.png') as drawer:
        options = {'1': main_1d,
                   '2': main_2d}
        run_option = input('For 1d perceptron enter 1, for 2d perceptron enter 2: ')
        options[run_option](drawer)
