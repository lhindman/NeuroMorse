import torch
import pickle
import random
import timeit
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn as nn
import snntorch as snn




#Define one layer spiking neural network for linear regression and test set evaluation.
class Net(nn.Module):
    def __init__(self,num_inputs,num_class, device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(num_inputs,num_class,bias = False)
        self.lif1 = snn.Leaky(beta = 0.95, reset_mechanism='zero',inhibition=True,learn_threshold=True,threshold = torch.ones(num_class))
        self.mem1 =torch.zeros(1,num_class,device=device)
        self.num_inputs = num_inputs
        self.num_class = num_class

        init_wt = torch.ones_like(self.fc1.weight.detach())

        init_thr = torch.ones_like(self.lif1.threshold.detach())
        self.STDP = []
        self.PosLength = 15
        self.NegLength = 15
        self.eta = 0.1
        self.Ath = 1e-1
        self.Tau_th = 1e-1/50
        self.idx_classification = 0
        with torch.no_grad():
            self.fc1.weight.copy_(init_wt)
            self.lif1.threshold.copy_(init_thr)

    def step(self, x):

        with torch.no_grad():
            cur1 = self.fc1(x) 
            print("DEBUG: Device for cur1:", cur1.device)
            print("DEBUG: Device for mem1:", self.mem1.device)
            spk1, self.mem1 = self.lif1(cur1.unsqueeze(0), self.mem1)

        return spk1,self.mem1
    def W1_Update(self,delta_t,spk1):
        vals = torch.where(delta_t>=0,delta_t, torch.tensor(self.PosLength, dtype=torch.long,device=self.device))
        vals.clamp_(0,self.PosLength).long()
        print("DEBUG: Device for vals:", vals.device)
        indices = vals + self.PosLength
        print("DEBUG: Device for indices:", indices.device)
        print("DEBUG: Device for self.STDP[1]:", self.STDP[1].device)
        STDP_w = torch.gather(self.STDP[1],0,torch.tensor(indices,dtype=torch.long,device=self.device)).repeat(self.num_class,1)
        delta_w = self.eta*(STDP_w - self.fc1.weight.detach())*spk1.transpose(0,1)
        NewWeights = self.fc1.weight.detach()+delta_w
        with torch.no_grad():
            self.fc1.weight.copy_(NewWeights.clamp(0,1))
    def PlotWeight(self,title):
        plt1 = plt.figure()
        plt.title('Receptive field')
        weights = self.fc1.weight.detach().cpu().numpy()
        plt.imshow(weights,vmin = 0, vmax = 1, cmap = "hot_r")  
        plt1.savefig(title)

def GenerateSTDP(PosLength,NegLength,Ap):
    # Generate STDP window
    
    DelT = torch.linspace(-NegLength,PosLength,NegLength+PosLength+1)
    WUpdate = torch.zeros(PosLength+NegLength+1)
    
    WUpdate[(NegLength):(PosLength + NegLength + 1)] = Ap*torch.linspace(1,0,PosLength+1)

    Window = torch.zeros((2,PosLength + NegLength+1))
    Window[0,:] = DelT
    Window[1,:] = WUpdate
    return Window
    #Note: using causal, weight dependant STDP rule from: "An Optimized Deep Spiking Neural Network Architecture Without Gradients"

def GenerateTrainSpikes(Dataset,num_classes = 50, shuffle = True, word_space = 15,num_channels = 2):
    #Use this to generate the data.
    #Also, should look at test set as well. This code should definitely be updated with some sort of transform and with batching.
    #TODO: Introduce a way to batch the data into smaller samples. Not really necessary for training, as training is small.
    SpikeData = []
    #Should randomise order for training set.
    for i in range(num_classes):
        data, label = Dataset[i]
        data_neuro = torch.zeros((int(data[-1][0])+1+word_space,num_channels))
    for idx in data: 
        data_neuro[int(idx[0]),int(idx[1])] = 1
    SpikeData.append(data_neuro)
    if shuffle == True:
        random.shuffle(SpikeData)
    return torch.cat(SpikeData,0) #consider removing torch.cat

def GenerateTestSpikes(TestSet, num_channels = 2, device = torch.device("cpu")):
    #Use this to generate the spikes for the test set.
    #compile spike sequence for test dataset
    #Below requires some form of batching or transform to handle more efficiently.
    #Consider if this is even necessary, wouldn't it be possible just to compare with the input each time and save memory?
    TestNeuro = torch.zeros((int(TestSet[0][-1][0])+1,num_channels), device=device)
    print("DEBUG: Device for TestNeuro:", TestNeuro.device)
    print("DEBUG: Device for TestSet:", TestSet[0].device)
    start_time = timeit.default_timer()
    counter = 0
    for idx in TestSet[0]:
        TestNeuro[int(idx[0]),int(idx[1])] = 1
        counter +=1
        if counter % 10000000 == 0:
            print('Time elapsed: %d, Counter = %d' %(timeit.default_timer() - start_time,counter))
    return TestNeuro

def Train(network,dataset,epochs,device = torch.device("cpu")):
    #Train network
    for epo in range(epochs):
        TrainSpikes = GenerateTrainSpikes(dataset,50,device=device)
        TrainSpikes = TrainSpikes.to(torch.float)

        #Convert each training input into spikes, and append into a list
        a = torch.randperm(network.num_class) #For random order

        input_times = torch.zeros(network.num_inputs) #Determines the most recent input.
        
        network.mem1.zero_()

        for t in range(TrainSpikes.shape[0]):
            spk1, mem1 = network.step(TrainSpikes[t,:])
            input_times[TrainSpikes[t,:]>0] = t

            #Update threshold activity
            with torch.no_grad():
                NewThresh = network.lif1.threshold.detach() - network.Tau_th + network.Ath*spk1.squeeze()
                network.lif1.threshold.copy_(NewThresh)

            if torch.sum(spk1)>0:
                delta_t = t - input_times
                network.W1_Update(delta_t,spk1)
                network.mem1.zero_()

    network.PlotWeight('Final Weight.png')
    return network

def Assign_Hidden_Layer(network,dataset, word_space = 15):
    #Determine which neuron is assigned to which keyword in the training set.
    #Assignment run
    #One additional run with no STDP or homeostatic regulation for class assignment.
    SpikeData = []


    for i in range(network.num_class):
        data, label = dataset[i]
        data_neuro = torch.zeros((int(data[-1][0])+1+word_space,network.num_inputs))
        for idx in data: 
            data_neuro[int(idx[0]),int(idx[1])] = 1
        SpikeData.append(data_neuro)

    Recorder = torch.zeros((network.num_class,network.num_class))
    i = 0
    for data in SpikeData:
        for t in range(data.shape[0]):
            spk1, mem1 = network.step(data[t,:])
            Recorder[i,:] += spk1.squeeze()
        i +=1

    #Using maximum spike count as a verification tool
    vals, idx_classification = torch.max(Recorder,dim=0) #idx_classification is numerical value of label.
    return idx_classification

def Test(network,dataset,idx_classification,device = torch.device("cpu")):
    #Test the network
    TestDict = dataset[1]
    TestingEndList = [] #List of list for all end times for each keyword

    for key in TestDict.keys():
        TestingEndList.append(TestDict[key][2])
    
    correct_spikes = 0
    incorrect_spikes = 0

    start_time = timeit.default_timer()

    #Not technically a confusion matrix, just a measure of correct vs incorrect for each class
    conf_matrix = torch.zeros((50,2))

    TestSpikes = GenerateTestSpikes(dataset,device=device)
    print("DEBUG: Device for TestSpikes:", TestSpikes.device)
    # TestSpikes = torch.ones((10000,2))
    # TestSpikes[200:300,0] = 1

    
    input_times = torch.zeros(network.num_inputs,device=device) #Determines the most recent input.
    print("DEBUG: Device for input_times:", input_times.device)
    for t in range(TestSpikes.shape[0]):
            spk1, mem1 = network.step(TestSpikes[t,:])
            input_times[TestSpikes[t,:]>0] = t

            if t%1000000 ==0 & t>0:
                print('Time Elapsed: %d, t = %d, correct = %i, incorrect = %i' %(timeit.default_timer()-start_time,t,correct_spikes,incorrect_spikes))

            if torch.sum(spk1)>0:
                spk1_label = idx_classification[spk1.nonzero()][0,1].item()
                if t in TestingEndList[spk1_label]:
                    correct_spikes +=1
                    conf_matrix[spk1_label,0] +=1
                else:
                    incorrect_spikes +=1
                    conf_matrix[spk1_label,1] +=1
                    
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

