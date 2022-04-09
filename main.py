from dots_drawer import DotsDrawer
from perceptron import *
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
        for i in range(3000):
            max_error = 0
            for dot, color in test_data:
                res = perc.eval(list(dot))
                err = (color - res) ** 2
                #print(f"{dot} -> {res:.4f} | {color} -> {err}")
                max_error = max(max_error, err)
                perc.teach(err=err, data=list(dot))
            print(f"{i: >4}: {max_error}")
            perc.learningRate = max_error * 0.001
            perc.print()
            # if max_error < 0.1:
            #     print(f"studied at {i}")
            #     break

        for _ in range(100):
            rand_dot = get_random_dot()
            if perc.eval(list(rand_dot)) > 0.5:
                drawer.dot(x=rand_dot[0], y=rand_dot[1], color=colours[1])
            else:
                drawer.dot(x=rand_dot[0], y=rand_dot[1], color=colours[0])

        for i in range(200):
            # Окей, тут норм кривая
            height = parabolize(sygmify((i-100) * 0.01)) * 50 + 200
            # print(f"d(s({i})) = {height}")
            drawer.dot(x=i, y=height, color=(128, 128, 32), size=2)
