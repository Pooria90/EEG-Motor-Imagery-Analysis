'''
This module containes the contains the codes that I wrote for my network trainings.
'''

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from copy import deepcopy

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

def update (model, opt, loss_func, xb, yb):
    '''
    function to do update model in one batch calculation
    
    model     : model that we want to update its weights in one epoch 
    opt       : optimizer
    loss_func : corresponding loss function
    xb, yb    : mini-batch and its labels
    '''
    
    model.train()
    opt.zero_grad()
    
    preds = model(xb)
    loss  = loss_func(preds, yb)
    acc   = accuracy(preds, yb)
    loss.backward()
    opt.step()
    
    return loss.item(), acc.item()


class EarlyStopping():
    def __init__(self, state, patience=10, attribute='loss'):
        '''
        a class for doining early stopping during training
        
        state     : use ES or not (boolean)
        patience  : if we see this number of results after our best result we break the training loop (int)
        attribute : the attribute for validation data that we decide the stopping based on that ('loss' or 'acc')
        '''
        
        self.state     = state
        self.patience  = patience
        self.attribute = attribute
        self.b_model   = nn.Module()                                # best model that is found during trainin
        self.atr_value = float('inf') if attribute == 'loss' else 0 # valid loss/acc of best model
        self.counter   = 0                                          # if counter==patience then stop training
        pass

    
def train (
    model, x_train, y_train, x_valid, y_valid,
    batch_size, epochs, learning_rate,
    loss_func = F.nll_loss, period = 1,
    er_stop = EarlyStopping(state=False)
    ):
    
    '''
    model         : the neural network that we want to train
    x_train       : training data (torch.tensor; dtype=float64)
    y_train       : training labels (torch.tensor; dtype=long)
    x_valid       : validation data (torch.tensor; dtype=float64)
    y_train       : validation labels (torch.tensor; dtype=long)
    batch_size    : mini-batch size for training (int)
    epochs        : number of training epochs (int)
    learning_rate : learning rate for optimizer (float)
    period        : period for printing training and validation logs (int)
    loss_func     : loss function that is used for updating weights
    device        : 'cpu' or 'gpu'
    er_stop       : EarlyStopping object (default is training loop without early-stopping)
    '''
    
    history = {'train_loss' : [],
               'train_acc'  : [],
               'valid_loss' : [],
               'valid_acc'  : []}
    
    if next(model.parameters()).is_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
        
    #train_ds = TensorDataset(x_train.float(), y_train.long())
    #train_dl = DataLoader(train_ds, batch_size=batch_size)
    
    train_ds = TensorDataset(x_train.float().to(device), y_train.long().to(device))
    train_dl = DataLoader(train_ds, batch_size=batch_size)
        
    opt = optim.Adam(model.parameters(),lr=learning_rate, eps=1e-8)
    
    for ep in range(1, epochs+1):
        if ep % period == 0 or ep == 1:
            print (f'\n*** Epoch: {ep} ***')
        
        tmp_loss, tmp_acc = [], []
        for xb, yb in train_dl:
            #xb, yb = x.to(device), y.to(device)
            loss, acc = update(model, opt, loss_func, xb, yb)
            tmp_loss.append(loss)
            tmp_acc. append(acc)
        
        history['train_loss'].append(sum(tmp_loss)/len(tmp_loss))
        history['train_acc' ].append(sum(tmp_acc)/len(tmp_acc))
        
        model.eval()
        with torch.no_grad():
            preds = model(x_valid.float().to(device))
            loss  = loss_func(preds,y_valid.long().to(device)).item()
            acc   = accuracy(preds, y_valid.long().to(device)).item()
        
        history['valid_loss'].append(loss)
        history['valid_acc'] .append(acc)
        
        if ep % period == 0 or ep == 1:
            print('Train Loss: {:.4f} --- Train Acc {:.2f}\nValid Loss: {:.4f} --- Valid Acc: {:.2f}'.format(
                history['train_loss'][-1], history['train_acc'][-1]*100,
                history['valid_loss'][-1], history['valid_acc'][-1]*100,
            ))
        
        if er_stop.state:
            if er_stop.attribute == 'loss':
                if history['valid_loss'][-1] < er_stop.atr_value:
                    er_stop.atr_value = history['valid_loss'][-1]
                    er_stop.b_model   = deepcopy(model)
                    er_stop.counter   = 0 
                else:
                    er_stop.counter  += 1
                    
            elif er_stop.attribute == 'acc':
                if history['valid_acc'][-1] > er_stop.atr_value:
                    er_stop.atr_value = history['valid_acc'][-1]
                    er_stop.b_model   = deepcopy(model)
                    er_stop.counter   = 0
                else:
                    er_stop.counter  += 1
            
            if er_stop.counter == er_stop.patience:
                model = deepcopy(er_stop.b_model)
                break
                
    return history
