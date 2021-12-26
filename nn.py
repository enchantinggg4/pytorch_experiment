import torch.nn

import pygad.torchga
import numpy


def create_network(input_size, output_size, num_solutions):
    input = torch.nn.Linear(input_size, input_size * 2)
    h1 = torch.nn.Linear(input_size * 2, input_size)
    r1 = torch.nn.ReLU()
    h2 = torch.nn.Linear(input_size, input_size)
    out = torch.nn.Linear(input_size, input_size)
    r2 = torch.nn.Softmax()

    model = torch.nn.Sequential(input, h1, r1, h2, out, r2)

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
