
import torch.nn
from nn import create_ga

import pygad.torchga
import numpy

(torch_ga, ga_instance) = create_ga()
ga_instance.run()

# In[39]:


# ga_instance.plot_result(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

solution, solution_fitness, solution_idx = ga_instance.best_solution()
best_solution_weights = pygad.torchga.model_weights_as_dict(model=torch_ga.model,
                                                            weights_vector=solution)
torch_ga.model.load_state_dict(best_solution_weights)
data_inputs = torch.tensor(numpy.random.random_sample((4))).float()
predictions = torch_ga.model(data_inputs)
print(data_inputs)
print('True : \n', data_inputs.prod().numpy())
print("Predictions : \n", predictions.detach().numpy())


torch.save(torch_ga.model, 'model.pt')
