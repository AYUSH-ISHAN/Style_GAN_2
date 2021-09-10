
from data_maker import data_generator
from main import training
from evaluation import final_evaluation

a = '''

Arguments of data_generator function :

size : size of image,
path : path to the lmbd file
num_of_workers : Number of worker s to allocate for multiprocessing
out : location to save the lmbd file format.

Arguments of training :

path : path of lmbd file
iterations : number of iteration 
batch : batch size of each iteration
checkpoint : location of last checkpoint

Arguments of final_evaluation :

path : path of the trained network to be feeded in generator model.

'''

print(a)

data_generator(size, path, num_of_workers, out, resample="lanczos")

training(path, iterations, batch_size, checkpoint, augment=True, wandb=True)

final_evaluation(path)
