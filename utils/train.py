import torch as T
import torch.nn as nn
#from torchtext import data, datasets
#from torchtext.vocab import Vocab
import torch.optim as optim
import time
import copy
import torch
import torch.nn.functional as F
from torchsummary import summary
import numpy as np




def train_one_epoch(model, dg_train, optimizer, loss_fn):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    mse_losses = []
    # Iterate over the DataLoader for training data
    for batch, (X,y) in enumerate(dg_train):
        
        Xt = torch.as_tensor(X).to(device)
        yt = torch.as_tensor(y).to(device)

        # Zero the gradients
        optimizer.zero_grad()
        
        pred = model(Xt).to(device)
        # compute loss 
        loss = loss_fn(pred, yt).to(device)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()
        
        mse_losses.append(loss.item())
        
    return(np.array(mse_losses).mean())


def validate_one_epoch(model, dg_valid, optimizer, loss_fn):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

      
    # Validation
    with torch.no_grad():
        mse_losses = []
        for batch, (X,y) in enumerate(dg_valid):
                validationStep_loss = []
                Xt = torch.as_tensor(X).to(device)
                yt = torch.as_tensor(y).to(device)

                # Zero the gradients
                optimizer.zero_grad()
                pred_val = model(Xt)
                # compute loss 
                validation_loss = loss_fn(pred_val, yt).to(device)
                   

                mse_losses.append(validation_loss.item())
                
        return(np.array(mse_losses).mean())