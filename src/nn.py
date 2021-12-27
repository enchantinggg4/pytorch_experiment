import torch.nn

import pygad.torchga
import numpy



class BattleModel(torch.nn.Module):


    def __init__(self, input_size):
        super().__init__()
        self.in_conv = torch.nn.Conv2d(2, 1, 1)
        self.input = torch.nn.Linear(input_size, input_size * 2)
        self.h1 = torch.nn.Linear(input_size * 2, input_size)
        self.r1 = torch.nn.ReLU()
        self.h2 = torch.nn.Linear(input_size, input_size)
        self.out = torch.nn.Linear(input_size, input_size)
        self.r2 = torch.nn.Softmax()

    def forward(self, x):
        x = self.in_conv(x)

        x = x.view(1, -1)

        x = self.input(x)
        x = self.h1(x)
        x = self.r1(x)
        x = self.h2(x)
        x = self.r2(x)
        x = self.out(x)
        x = self.r2(x)
        return x.view(-1)



def create_network(input_size, output_size, num_solutions):
    model = BattleModel(input_size)

    return pygad.torchga.TorchGA(model=model, num_solutions=num_solutions)

def create_ga(input_size, iters, fitness_func, num_solutions=10, callback_gen=None):


    # In[37]:

    input_count = input_size
    out_count = 2

    torch_ga = create_network(input_count, out_count, num_solutions)

    # Creating the initial population.

    num_generations = iters
    num_parents_mating = 4

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"
    initial_population = torch_ga.population_weights

    mutation_type = "random"
    mutation_percent_genes = 10
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           keep_parents=keep_parents,
                           on_generation=callback_gen
                           )
    return torch_ga, ga_instance
