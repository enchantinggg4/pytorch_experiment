import random

import numpy
import torch
import pygad
import pygad.torchga
from nn import create_ga
import math
import json

speed = 4



# (torch_ga, ga_instance) = create_ga(64, 1, fitness_func, 10, callback_gen)


def create_ok_solution(field):
    my_idx = [idx for (idx, x) in enumerate(field) if x == 1][0]
    x, y = my_idx % 8, my_idx // 8

    enemies = [
        (idx % 8, idx // 8) for (idx, x) in enumerate(field) if x == -1
    ]

    closest = None
    pd = 1000000
    for (ex, ey) in enemies:
        dist = math.sqrt((ex - x) ** 2 + (ey - y) ** 2)
        if dist < pd:
            closest = (ex, ey)
            pd = dist

    tx, ty = closest

    dx, dy = tx - x, ty - y

    spd = field[-1]
    if pd <= spd:
        # we can hit it is in distance
        return dx, dy

    # dx, dy = (dx / pd) * spd, (dy / pd) * spd
    return dx, dy


def create_random_case():
    field = []
    # initial field
    for y in range(8):
        for x in range(8):
            field.append(0)

    # ai's pawn

    for i in range(1):
        field[random.randint(0, 63)] = 1

    # enemy pawn
    for i in range(3):
        spot = random.randint(0, 63)
        if field[spot] == 0:
            field[spot] = -1

    field.append(speed)

    solution = create_ok_solution(field)

    for x in range(0, 64):
        if field[x] == 1:
            print('▣', end='  ')
        elif field[x] == -1:
            print('⬤', end='  ')
        else:
            print(field[x], end='  ')
        if x % 8 == 0:
            print()
    print()

    print('Solution', solution)

    return [field, solution]


def create_data_set(size):
    dataset = []
    for i in range(size):
        dataset.append(create_random_case())

    f = open("dataset.json", "w")
    f.write(json.dumps(dataset))
    f.close()


create_data_set(5000)





# dataset = []

with open('dataset.json') as json_file:
    dataset = json.load(json_file)

def callback_gen(ga_instance):
    print("%d Fitness of the best solution :" % ga_instance.generations_completed, ga_instance.best_solution()[1])


loss_function = torch.nn.L1Loss()




def train():
    input_data = torch.tensor([x[0] for x in dataset]).float()
    out_data = torch.tensor([x[1] for x in dataset]).float()
    def fitness_func(solution, solution_idx):
        model_weights_dict = pygad.torchga.model_weights_as_dict(model=torch_ga.model,
                                                                 weights_vector=solution)
        # Use the current solution as the model parameters.
        torch_ga.model.load_state_dict(model_weights_dict)

        predictions = pygad.torchga.predict(model=torch_ga.model,
                                            solution=solution,
                                            data=input_data)

        abs_error = loss_function(predictions, out_data).detach().numpy() + 0.00000001

        solution_fitness = 1.0 / abs_error
        return solution_fitness


    input_size = 8 * 8
    (torch_ga, ga_instance) = create_ga(input_size, 3000, fitness_func, 15, callback_gen)
    ga_instance.run()


train()