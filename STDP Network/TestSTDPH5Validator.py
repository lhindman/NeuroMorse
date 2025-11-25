# import torch
import dill as pickle
import argparse
import torch
from STDPNetwork import Net
from STDPNetwork import Test
from micronets import NeuroMorseDataset

def run(network_filename, dataset_filename):
    f = open(network_filename,'rb')
    network = pickle.load(f)
    f.close()

    test_ds = NeuroMorseDataset(dataset_filename, dt_us=1)

    # Set the seed for reproducibility
    torch.manual_seed(42)


    Test(network,test_ds,network.idx_classification)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_filename",type=str,default = 'Network.pckl')
    parser.add_argument("--dataset_filename",type = str, default = './data/Clean-Test.h5')
 
    args = parser.parse_args()
    
    run(**vars(args))