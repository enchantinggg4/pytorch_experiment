import os
import sys
import time

import torch
import pygad

from mygym import GymTester, create_gym, AI

model = torch.load('models/100_84.pt')

ga = pygad.torchga.TorchGA(model = model, num_solutions=10)

g = GymTester(10)



# test()


bar = [
    " [=     ]",
    " [ =    ]",
    " [  =   ]",
    " [   =  ]",
    " [    = ]",
    " [     =]",
    " [    = ]",
    " [   =  ]",
    " [  =   ]",
    " [ =    ]",
]


def test(side = False):
    ais = [AI(x) for x in [ga, ga]]


    if side:
        ais = ais[::-1]  # reverse so tested ai becomes second(right side)
    g = create_gym(ais[0], ais[1])

    state_arr = []
    for i in range(15):
        state = g.turn(True, dprint=False)
        state_arr += state

    for idx, state in enumerate(state_arr):
        os.system('cls' if os.name=='nt' else 'clear')
        print('State %d' % idx)
        print(state)
        sys.stdout.flush()
        time.sleep(0.5)

    return g

test()