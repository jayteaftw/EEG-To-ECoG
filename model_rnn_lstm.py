import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import torch 
from scipy.io import loadmat
import numpy as np
import time
import math
import matplotlib.pyplot as plt


def min_max_scale(A,a,b):
    return a + (A - np.min(A))*(b-a)/(np.max(A) - np.min(A))

def reverse_min_max_scale(X,minA,maxA,a,b):
    return minA + ((X-a)*(maxA - minA)/(b-a))

def preprocess(EEG,ECoG,time_step,a,b,input_dim,output_dim):
    
    #EEG /= 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EEG = torch.tensor(np.transpose(EEG), dtype=torch.float).to(device)
    saved_ECoG = torch.tensor(np.transpose(ECoG), dtype=torch.float).to(device)    
    ECoG = torch.tensor(np.transpose(ECoG), dtype=torch.float).to(device)
    
    print("here: ",saved_ECoG[0][0].item(),ECoG[0][0].item())
    
    _, _, V = torch.pca_lowrank(ECoG,q=output_dim)
    print(V.shape)
    ECoG = torch.matmul(ECoG, V[:, :output_dim])
    saved_ECoG = torch.matmul(saved_ECoG, V[:, :output_dim])


    _, _, V = torch.pca_lowrank(EEG,q=input_dim)
    print(V.shape)
    EEG = torch.matmul(EEG, V[:, :input_dim])
    
    
    

    EEG = EEG.numpy(force=True)
    ECoG = ECoG.numpy(force=True)

    ECoG_Max = np.max(ECoG)
    ECoG_Min = np.min(ECoG)

    ECoG = min_max_scale(ECoG,a,b)
    EEG = min_max_scale(EEG,a,b)

    EEG = torch.tensor(EEG, dtype=torch.float).to(device)    
    ECoG = torch.tensor(ECoG, dtype=torch.float).to(device)
    saved_ECoG = torch.tensor(saved_ECoG, dtype=torch.int)

    print("here: ",saved_ECoG[0][0].item(), reverse_min_max_scale(ECoG[0][0].item(),ECoG_Min,ECoG_Max,a,b ))
    
    print(ECoG.shape)
    print("PCA", ECoG.shape)
    print(EEG.shape)
    print(EEG.dtype, ECoG.dtype)
    batches = 1
    batch_size = int(EEG.shape[0] // time_step //batches)
    EEG = EEG.reshape((batches, batch_size, time_step, EEG.shape[1]))
    ECoG = ECoG.reshape((batches, batch_size, time_step, ECoG.shape[1]))
    saved_ECoG = saved_ECoG.reshape((batches, batch_size, time_step, saved_ECoG.shape[1]))
    
    print("here3: ",saved_ECoG[0][0][0][0].item(), reverse_min_max_scale(ECoG[0][0][0][0].item(),ECoG_Min,ECoG_Max,a,b ))
    print(EEG.shape, ECoG.shape)
    dataset = TensorDataset(EEG,ECoG)


    val_size = 1000
    test_size = 2000
    training_data = (EEG[:,val_size+test_size:,:],ECoG[:,val_size+test_size:,:])
    testing_data = (EEG[:,val_size:val_size+test_size,:],ECoG[:,val_size:val_size+test_size,:])
    validation_data = (EEG[:,:val_size,:],ECoG[:,:val_size,:])
    saved_ECoG = saved_ECoG[:,val_size+test_size:,:]
    train_size = training_data[0].shape[1]
    print(saved_ECoG[0][200][0][0].item(), reverse_min_max_scale(training_data[1][0][200][0][0].item(),ECoG_Min,ECoG_Max,1,-1 ))
    #print(hidden_size)
    return training_data, validation_data, testing_data, val_size, train_size, saved_ECoG, ECoG_Min, ECoG_Max


if __name__ == "__main__":
    EEG = loadmat('Data/EEG_rest.mat')['EEG']
    ECoG = loadmat('Data/ECoG_rest.mat')['ECoG']
    print(EEG.shape) 
    print(ECoG.shape)
    print(type(EEG[0][0]), type(ECoG[0][0]))
    ECoG = ECoG.astype('float')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dim = 16
    input_dim = 16
    a,b = 1,-1
    time_step = 50
    ECoG_Max = np.max(ECoG)
    ECoG_Min = np.min(ECoG)
    #Standarize Data
    #ECoG = (ECoG - np.mean(ECoG))/np.std(ECoG)

    training_data, validation_data, testing_data, val_hidden_size, train_hidden_size, saved_ECoG, ECoG_Min, ECoG_Max = preprocess(EEG,ECoG,time_step, a,b,input_dim,output_dim)
    print(training_data[0].shape,validation_data[0].shape,testing_data[0].shape )
    
    num_layers = 2
    model_type = "LSTM"
    if model_type == "RNN":
        model = nn.RNN(input_size=input_dim, hidden_size=input_dim, num_layers=num_layers, batch_first=True).to(device)
    else:
        model = nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=num_layers, batch_first=True).to(device)
    
    crit = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    total_epochs = 500
    
    print("saved, ecog ",saved_ECoG.shape, training_data[1].shape )
    def train_model(total_epochs, print_every=100):
        prev_loss = 0
        all_loss = []
        val_losses = []
        epochs_axis = []
        final_output = []
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(total_epochs):
            model.train()
            running_loss = 0.0
            hidden = torch.randn(num_layers, train_hidden_size, input_dim).to(device)
            cell = torch.randn(num_layers, train_hidden_size, input_dim).to(device)
            epoch_loss = 0
            inputs, labels = training_data
            for i, input in enumerate(inputs):
                if model_type == "RNN":
                    output, hidden = model(input, hidden)
                else:
                    output, (hidden,cell) = model(input, (hidden, cell))
                
            final_output = output
            optim.zero_grad()
            loss = crit(output, labels[0])
            loss.backward()
            optim.step()
            running_loss += loss.item()
            epoch_loss += loss.item()

            if (epoch) % 20 == 0:

                model.eval()
                val_input, val_labels = validation_data
                hidden = torch.randn(num_layers, val_hidden_size, input_dim).to(device)
                cell = torch.randn(num_layers, val_hidden_size, input_dim).to(device)
                for i, input in enumerate(val_input):
                    if model_type == "RNN":
                        output, hidden = model(input, hidden)
                    else: 
                        output, (hidden,cell) = model(input, (hidden, cell))
                loss = crit(output, val_labels[0])
                val_loss = loss.item()

                channel = 9
                step = 10
                batch_step = 100
                print(f'[e:{epoch}/{total_epochs}, i:{i + 1:5d}/{len(EEG)}] loss: {running_loss/100:.4f} \t \n \
                        inputs:{np.array(inputs.cpu().detach().numpy())[0][batch_step][step][channel]} \n\
                        output:{reverse_min_max_scale(np.array(output.cpu().detach().numpy())[batch_step][step][channel],ECoG_Min,ECoG_Max,a,b )} \n\
                        label:{reverse_min_max_scale(np.array(labels.cpu().detach().numpy())[0][batch_step][step][channel],ECoG_Min,ECoG_Max,a,b )}\n \
                        actual_label:{np.array(saved_ECoG.cpu().detach().numpy())[0][batch_step][step][channel]}')
                #print(outputs.shape, labels.shape)
                running_loss = 0.0
                print(f'epoch {epoch} train loss: {epoch_loss} Decrease In Loss: {prev_loss - epoch_loss}  \n\t validation loss {val_loss}\n ')
                prev_loss = epoch_loss
                all_loss.append(epoch_loss)
                val_losses.append(val_loss)
                epochs_axis.append(epoch)
        print(all_loss)
        print(val_losses)
        print(epochs_axis)


        #eval
        print("Eval")
        model.eval()
        train_input, train_labels = validation_data
        hidden = torch.randn(num_layers, val_hidden_size, input_dim).to(device)
        cell = torch.randn(num_layers, val_hidden_size, input_dim).to(device)
        for i, input in enumerate(train_input):
            if model_type == "RNN":
                output, hidden = model(input, hidden)
            else: 
                output, (hidden,cell) = model(input, (hidden, cell))
        loss = crit(output, train_labels[0])
        val_loss = loss.item()
        final_output = output

        

        

        final_output = reverse_min_max_scale(final_output.cpu().detach().numpy().reshape(-1,16),ECoG_Min,ECoG_Max,a,b)
        labels = reverse_min_max_scale(train_labels.cpu().detach().numpy().reshape(-1,16),ECoG_Min,ECoG_Max,a,b)
        inputs = train_input.cpu().detach().numpy().reshape(-1,16)
        print(final_output.shape, labels.shape, inputs.shape)

        np.save(f"{model_type}_train_loss.npy",all_loss)
        np.save(f"{model_type}_val_loss.npy",val_losses)
        np.save(f"{model_type}_epochs_axis.npy",epochs_axis)
        np.save(f"{model_type}_output.npy",final_output)
        np.save(f"{model_type}_labels.npy",labels)

        

    train_model(total_epochs)