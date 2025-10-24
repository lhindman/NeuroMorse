# import torch
import dill as pickle
import argparse
from STDPNetwork import Net
from STDPNetwork import Test

def run(network_filename, dataset_filename):
    f = open(network_filename,'rb')
    network = pickle.load(f)
    f.close()

    f = open(dataset_filename,'rb')
    dataset = pickle.load(f)
    f.close()

    print("DEBUG: Network device:", network.device)

    Test(network,dataset[0],network.idx_classification,device=network.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_filename",type=str,default = 'Network.pckl')
    parser.add_argument("--dataset_filename",type = str, default = './data/Test_Dropout-None_Jitter-None_Poisson-None.pckl')
 
    args = parser.parse_args()
    
    run(**vars(args))