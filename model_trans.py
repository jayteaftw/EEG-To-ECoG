import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import torch 
from scipy.io import loadmat
import numpy as np
import time
import math




class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        output = x + self.pe[:x.size(0)]
        #print(x.size(), output.size())
        return self.dropout(output)

""" class MyTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=10, nlayers=12, dropout=0.5,input=16,output=128):
        super(MyTransformer, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc = nn.Linear(input,output)
    
    def forward(self, src):
        #src = self.pos_encoder(src)
        x = self.transformer_encoder(src)
        return self.fc(x) """

class IntifyScaled(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """ print(x[0][0][0])
        x *= 100
        x = x.int()
        x = x.float()
        print("new out", x.shape)
        print(x[0][0][0]) """
        #x = 1000*x
        return x.int().float().requires_grad_()


class MyTransformer(nn.Module):
    def __init__(self, output: int, d_model: int, nhead: int, d_hid: int=2048, nlayers: int=12, dropout: float = 0.2):  
        super().__init__()
        #super(MyTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, output)
        #self.init_weights()


    def init_weights(self) -> None:
        initrange = .1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        
        """ Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken] """
       

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        #output = self.intify(output)
        return output



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
    
    #Standarize Data
    ECoG = (ECoG - np.mean(ECoG))/np.std(ECoG)
    EEG = (EEG - np.mean(EEG))/np.std(EEG) 


    #EEG /= 1
    EEG = torch.tensor(np.transpose(EEG), dtype=torch.float).to(device)    
    ECoG = torch.tensor(np.transpose(ECoG), dtype=torch.float).to(device)

    
    _, _, V = torch.pca_lowrank(ECoG,q=output_dim)
    print(V.shape)
    ECoG = torch.matmul(ECoG, V[:, :output_dim])


    _, _, V = torch.pca_lowrank(EEG,q=input_dim)
    print(V.shape)
    EEG = torch.matmul(EEG, V[:, :input_dim])
    


    print(ECoG.shape)
    print("PCA", ECoG.shape)
    print(EEG.shape)
    print(EEG.dtype, ECoG.dtype)
    time_step = 20
    batch_size = 1000
    batches = int(int(EEG.shape[0] / time_step) /batch_size)
    EEG = EEG.reshape((batches, batch_size, time_step, EEG.shape[1]))
    ECoG = ECoG.reshape((batches, batch_size, time_step, ECoG.shape[1]))
    print(EEG.shape, ECoG.shape)
    dataset = TensorDataset(EEG,ECoG)
    print(len(dataset[0]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
   
    
    model = MyTransformer(output=output_dim, d_model = input_dim, nhead=input_dim, nlayers=12).to(device)
    crit = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    total_epochs = 100

    #outputs = model(EEG[0])
    all_loss = []
    def train_model(total_epochs, print_every=100):
        
        prev_loss = 0
        for epoch in range(total_epochs):
            
            running_loss = 0.0
            epoch_loss = 0
            for i, data in enumerate(zip(EEG,ECoG)):
                
                inputs, labels = data
                #inputs = np.array([inputs])
                #print("input/label: ",inputs.shape, labels.shape)
                #inputs, labels = np.transpose(inputs), np.transpose(labels)
                #print(inputs.shape, labels.shape)
                optim.zero_grad()
                outputs = model(inputs)
                #print("output:", outputs.shape)
                #print("input size:", inputs.size(), "dataloader:",len(dataloader))
                #print("output size:", outputs.size(), ", label size:", labels.size())
                #print(outputs.view(-1, 128).shape)
                loss = crit(outputs, labels)
                loss.backward()
                optim.step()

                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 10 == 0:
                    #print("output:", outputs.shape, "labels:",labels.shape)
                    print(f'[e:{epoch + 1}/{total_epochs}, i:{i + 1:5d}/{len(EEG)}] loss: {running_loss/100:.4f} \t output:{np.array(outputs.cpu().detach().numpy())[0][0][0]} label:{np.array(labels.cpu().detach().numpy())[0][0][0]}')
                    running_loss = 0.0
            print(f'epoch {epoch} loss: {epoch_loss/len(EEG)} Decrease In Loss: {prev_loss - epoch_loss}\n')
            prev_loss = epoch_loss
            all_loss.append(epoch_loss/len(EEG))
        print(all_loss)
        torch.save(model.state_dict(), "model.pt")
    train_model(total_epochs)
