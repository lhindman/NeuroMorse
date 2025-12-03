# import torch
import dill as pickle
import timeit
import torch
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from STDPNetwork import Net
from micronets import NeuroMorseDataset

def Test(network,dataset,idx_classification):
    #Test the network
    # print('Beginning Test Set Evaluation')
    # print(" The dataset has type: %s" % type(dataset))
    # print(" The type of the first element is: %s" % type(dataset[0]))
    # print("THe first elemen is: %s" % dataset[0])
    # TestDict = dataset[1]
    # TestingEndList = [] #List of list for all end times for each keyword

    # for key in TestDict.keys():
    #     TestingEndList.append(TestDict[key][2])
    
    correct_spikes = 0
    incorrect_spikes = 0

    start_time = timeit.default_timer()

    #Not technically a confusion matrix, just a measure of correct vs incorrect for each class
    conf_matrix = torch.zeros((50,2))

    # The NeuroMorseDataset test mode provides direct access to spike trains and labels, 
    #   however it contains only a single trial per call. As a result we specifically retrieve the first item.
    TestSpikes, label = dataset[0]

    # TestSpikes = GenerateTestSpikes(dataset)
    # TestSpikes = torch.ones((10000,2))
    # TestSpikes[200:300,0] = 1

    
    input_times = torch.zeros(network.num_inputs)
    print("Beginning Test Set Evaluation")
    for t in range(TestSpikes.shape[0]):
        spk1, mem1 = network.step(TestSpikes[t,:])
        input_times[TestSpikes[t,:]>0] = t

        if t % 1000 == 0:
            print('Time Elapsed: %d, t = %d, correct = %i, incorrect = %i' %(timeit.default_timer()-start_time,t,correct_spikes,incorrect_spikes))
            print("Completed = %0.2f%%" %(((t/TestSpikes.shape[0]))*100))

        if torch.sum(spk1)>0:
            spk1_label = idx_classification[spk1.nonzero()][0,1].item()
            
            _, TestingEndList = dataset.get_test_times(spk1_label)
            # Similiar to above, since only one trial is present, we access the first element directly.
            if t in TestingEndList[0]:
                correct_spikes +=1
                conf_matrix[spk1_label,0] +=1
            else:
                incorrect_spikes +=1
                conf_matrix[spk1_label,1] +=1
        # print("Completed %d/%d timesteps" %(t+1,TestSpikes.shape[0]))
    print("Test Set Evaluation Complete")
    print('Correct Spikes')
    print(correct_spikes)
    print('Incorrect Spikes')
    print(incorrect_spikes)
    print('Confusion Matrix')
    print(conf_matrix)

    plt.figure(figsize = (15,10))
    img = sns.heatmap(
            conf_matrix,annot= True, cmap = "YlGnBu",cbar_kws= {"label":"Scale"}
        )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.savefig('Confusion Matrix.svg') #1st row is correct spikes, 2nd row is incorrect spikes
    plt.close()


def GenerateTestSpikes(TestSet, num_channels = 2):
    #Use this to generate the spikes for the test set.
    #compile spike sequence for test dataset
    #Below requires some form of batching or transform to handle more efficiently.
    #Consider if this is even necessary, wouldn't it be possible just to compare with the input each time and save memory?
    print("Generating Test Spikes")
    TestNeuro = torch.zeros((int(TestSet[0][-1][0])+1,num_channels))
    start_time = timeit.default_timer()
    counter = 0
    total_idx = len(TestSet[0])
    for idx in TestSet[0]:
        TestNeuro[int(idx[0]),int(idx[1])] = 1
        counter +=1
        if counter % 1000 == 0:
            print('Time elapsed: %d, Completed = %0.2f%%' %(timeit.default_timer() - start_time,(counter/total_idx)*100))
    print("Finished Generating Test Spikes")
    return TestNeuro


def run(network_filename, dataset_filename):
    f = open(network_filename,'rb')
    network = pickle.load(f)
    f.close()

    test_ds = NeuroMorseDataset(dataset_filename, dt_us=1, test_mode=True)

    # Set the seed for reproducibility
    torch.manual_seed(42)


    Test(network,test_ds,network.idx_classification)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_filename",type=str,default = 'Network.pckl')
    parser.add_argument("--dataset_filename",type = str, default = './data/Clean-Test.h5')
 
    args = parser.parse_args()
    
    run(**vars(args))