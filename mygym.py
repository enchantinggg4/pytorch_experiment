import random

import numpy
import torch
import pygad
import pygad.torchga
from nn import create_ga, create_network
import math


class Gym:

    def __init__(self, w, h, left_ai, right_ai):
        self.turn_i = 0
        self.w = w
        self.h = h
        self.left_ai = left_ai
        self.right_ai = right_ai
        self.left_units = []
        self.right_units = []

    def prepare(self):
        i = 0
        for unit in self.left_units:
            unit.set_position(0, i)
            unit.side = True
            i += random.randint(1, 2)
        i = 0
        for unit in self.right_units:
            unit.set_position(self.w - 1, i)
            unit.side = False
            i += random.randint(1, 2)

    def turn(self, represent, dprint = True):
        # for now - left to right foreach

        states = []

        for unit in self.left_units:
            self.action(
                self.left_ai,
                unit,
                self.left_ai.decide(unit,
                                    self)
            )
            if represent:
                states.append(self.represent(dprint))


        for unit in self.right_units:
            self.action(
                self.right_ai,
                unit,
                self.right_ai.decide(unit,
                                     self)
            )
            if represent:
                states.append(self.represent(dprint))
        self.turn_i += 1
        return states

    def action(self, ai, unit, act):
        act = numpy.array(act)
        pos = numpy.array([unit.x, unit.y])

        new_pos = (act - pos)
        x_diff = round(new_pos[0])
        y_diff = round(new_pos[1])

        new_x = unit.x + x_diff
        new_y = unit.y + y_diff

        dist = math.sqrt(x_diff ** 2 + y_diff ** 2)

        if self.is_out_of_bounds(new_x, new_y) or dist > unit.speed:
            ai.score -= 1
        elif self.is_occupied(new_x, new_y) and self.get_unit_at(new_y, new_y) != unit:
            ai.score -= 1
        else:
            unit.set_position(new_x, new_y)
            e = self.get_adjacent_enemy(new_x, new_y, unit.side)
            if e is not None:
                dmg = unit.damage * unit.count
                fh = (e.count - 1) * e.max_health + e.health
                fh -= dmg
                new_c = fh // e.max_health
                new_h = fh  % e.max_health
                e.count = new_c
                e.health = new_h
                if new_c <= 0:
                    [self.right_units, self.left_units][e.side].remove(e)
                    ai.score += 50
                ai.score += dmg
                opp_ai = self.right_ai if self.left_ai == ai else self.left_ai
                opp_ai.score -= unit.damage


            # here we check if enemy adjacent

    def get_adjacent_enemy(self, x, y, side):
        adj_mat = [
            [-1, 0],  # L
            [-1, -1],  # TL
            [0, -1],  # T
            [1, -1],  # TR
            [1, 0],  # R
            [1, 1],  # BR
            [0, 1],  # B
            [-1, 1],  # BL
        ]

        for adj in adj_mat:
            pos = numpy.array([x, y]) + adj

            if self.is_out_of_bounds(*pos):
                continue

            if self.is_occupied(*pos):
                u = self.get_unit_at(*pos)
                if u.side != side:
                    return u

        return None

    def is_out_of_bounds(self, x, y):
        return x >= self.w or y >= self.h or x < 0 or y < 0

    def is_occupied(self, x, y):
        return self.get_unit_at(x, y) is not None

    def get_unit_at(self, x, y):
        for unit in self.left_units + self.right_units:
            if unit.x == x and unit.y == y:
                return unit
        return None

    def represent(self, dprint = True):
        table = []
        for x in range(self.w):
            arr = []
            table.append(arr)
            for y in range(self.h):
                arr.append('-')

        # default, 1st unit, 2nd unit in team
        colors = ['\033[91m', '\033[94m', '\033[92m', '\033[93m', '\033[95m', '\033[96m']

        i = 1
        for unit in self.left_units:
            table[unit.y][unit.x] = colors[i] + '▣' + colors[0]
            i += 1
        i = 1

        for unit in self.right_units:
            table[unit.y][unit.x] = colors[i] + '△' + colors[0]
            i += 1


        t = ''
        if dprint:
            print('State, turn ', self.turn_i)
        for x in range(self.w):
            for y in range(self.h):
                if dprint:
                    print(table[x][y], end='  ')
                t += table[x][y] + '  '
            if dprint:
                print()
            t += '\n'
        return t


class Unit:

    def __init__(self, speed, damage, health, count):
        self.speed = speed
        self.damage = damage
        self.count = count
        self.max_health = health
        self.health = health
        self.side = True
        self.x = -1
        self.y = -1

    def set_position(self, x, y):
        self.x = x
        self.y = y


class AI:
    def __init__(self, model):
        self.model = model
        self.score = 0

    def get_input(self, unit, gym):
        i = numpy.zeros(8 * 8).reshape((8, 8))
        for u in gym.left_units + gym.right_units:
            # index = u.x + u.y * gym.w
            if unit.side == u.side:
                i[u.y][u.x] = -1
            else:
                i[u.y][u.x] = 1

        return i.flatten()

    def decide(self, unit, gym):
        # inputs
        input = self.get_input(unit, gym)
        # outputs
        # 1x64 array of weights
        decision = self.model.model(torch.tensor(input).float())
        # we need to find in range best spot
        bw = -1
        bwi = -1
        for (idx, weight) in enumerate(decision):
            px, py = idx % 8, idx // 8
            dist = math.sqrt((px - unit.x) ** 2 + (py - unit.y) ** 2)
            if weight > bw and dist <= unit.speed:
                bw = weight
                bwi = idx

        return [bwi % 8, bwi // 8]


def create_gym(left_ai, right_ai):
    g = Gym(8, 8, left_ai, right_ai)
    speed = 4
    damage = 15
    health = 20

    min, max = 3,3

    for i in range(random.randint(min, max)):
        g.left_units.append(
            Unit(speed, damage, health, random.randint(3, 5))
        )
    for i in range(random.randint(min, max)):
        g.right_units.append(
            Unit(speed, damage, health, random.randint(3, 5))
        )
    g.prepare()
    return g


def save_model(model, filename):
    torch.save(model, filename)


class GymTester:

    def run_simulation(self, ai_to_test, solution, side=False, rep=False):
        # get weights from current model
        model_weights_dict = pygad.torchga.model_weights_as_dict(model=ai_to_test.model, weights_vector=solution)
        # Use the current solution as the model parameters.
        ai_to_test.model.load_state_dict(model_weights_dict)
        # create gym where left is tested model and right is previous generation

        ais = [AI(ai_to_test), AI(ai_to_test) if self.previous_best_model is None else AI(self.previous_best_model)]
        if side:
            ais = ais[::-1] # reverse so tested ai becomes second(right side)
        g = create_gym(ais[0], ais[1])
        for i in range(15):
            g.turn(rep)

        return g

    def callback_gen(self, ga_instance):
        model = self.previous_best_model.model
        weights = pygad.torchga.model_weights_as_dict(model=model, weights_vector=self.ga_instance.best_solution()[0])
        self.previous_best_model.model.load_state_dict(weights)
        print("%d Fitness of the best solution :" % ga_instance.generations_completed, ga_instance.best_solution()[1])

    def fitness_func(self, solution, solution_idx):
        # random_side = bool(random.getrandbits(1))
        random_side = False
        g = self.run_simulation(self.training_ga, solution, side=random_side, rep = False)


        score = g.left_ai.score if random_side == False else g.right_ai.score
        return score

    def __init__(self, iterations_to_run):
        self.iterations = iterations_to_run
        self.input_size = 8 * 8
        self.training_ga, self.ga_instance = create_ga(self.input_size, iterations_to_run,
                                                       lambda x, y: self.fitness_func(x, y), 10,
                                                       callback_gen=lambda instance: self.callback_gen(instance))
        self.previous_best_model = create_network(self.input_size, 64, 10)
        pass

    def run(self):
        self.ga_instance.run()

        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()

        save_model(self.training_ga.model, 'models/%d_%d.pt' % (self.iterations, solution_fitness))


        self.previous_best_model = self.training_ga
        self.run_simulation(self.training_ga, solution, side=False, rep=True)

        # self.ga_instance.plot_result(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# GymTester(2500).run()
