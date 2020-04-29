
import os
import random
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import get_dataloader
from models import InteractionNetwork 
from hyperparams import *
from data_preprocessing import binary_indices, mse_indices, categorical_indices, index_mapping

from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()

data = np.load(os.path.join("data","featurized_train_24.npy"))

print("Data shape: ", data.shape) 
train_dataloader, validation_dataloader = get_dataloader(data, batch_size, USE_CUDA, object_dim, n_objects, relation_dim)



interaction_network = InteractionNetwork(n_objects, object_dim, n_relations, relation_dim, effect_dim)
device = "cpu"
if USE_CUDA:
    interaction_network = interaction_network.cuda()
    device= "cuda"
optimizer = optim.Adam(interaction_network.parameters())
#criterion = nn.MSELoss()



def validate(validation_loader, model):
    losses = []
    for dp, target in validation_loader:
        objects, sender_relations, receiver_relations, relation_info = dp
        predicted = interaction_network(objects, sender_relations, receiver_relations, relation_info)
        calculate_total_loss(predicted, target)
        #loss = criterion(predicted, target)  
        losses.append(loss.item())
    return np.mean(losses)


def mae(predicted, target):
    diff = torch.abs(predicted - target)
    diff = diff.reshape(-1, predicted.shape[-1])
    mae_mean = diff.mean(dim=0)
    return mae_mean

def add_vars(writer, diff, epoch):
    writer.add_scalar("presence bit", diff[0], epoch)
    writer.add_scalar("Occluder bit: ", diff[1], epoch)
    for key, val in index_mapping.items():
        writer.add_scalar(key, diff[val], epoch)



n_epoch = 200
batches_per_epoch = 100
loss_methods = {k: nn.MSELoss(reduction='mean') for k in mse_indices}
loss_methods.update({k: nn.CrossEntropyLoss(reduction='mean') for k in binary_indices})
loss_methods.update({k: nn.CrossEntropyLoss(reduction='mean') for k in categorical_indices})


def calculate_total_loss(predicted, target):

    # criterion = nn.MSELoss(reduction='mean')
    # return criterion(predicted, target)

    loss = None

    for key, val in index_mapping.items():
        
        cur_criterion = loss_methods[key]
        cur_target = target[:,:,val[0]:val[-1]+1].reshape(-1,len(val))
        if "CrossEntropyLoss" in str(cur_criterion):
            _, cur_target = cur_target.max(dim=1)
            cur_target = cur_target.long()

        cur_predicted = predicted[:,:,val[0]:val[-1]+1].reshape(-1, len(val))
        cur_loss = cur_criterion(cur_predicted, cur_target)
        
        if loss is None:
            loss = cur_loss
        else:
            loss +=  cur_loss
    return loss


all_train_losses = []
all_valid_losses = []
best_val = 100000
for epoch in tqdm(range(1,n_epoch+1)):
    losses = []
    running_mae = torch.zeros(data.shape[-1]).to(device)
    for dp, target in train_dataloader:
        objects, sender_relations, receiver_relations, relation_info = dp
        predicted = interaction_network(objects, sender_relations, receiver_relations, relation_info)
        mae_mean = mae(predicted, target)
        running_mae += mae_mean
        
        #loss = criterion(predicted, target)
        loss = calculate_total_loss(predicted, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    running_mae /= len(train_dataloader)
    # add_vars(writer, running_mae, epoch)
    train_loss = np.mean(losses)
    valid_loss = validate(validation_dataloader, interaction_network)  
    writer.add_scalars("Losses", {'train_mse': train_loss, 'valid_mse' : valid_loss}, epoch)
    
    all_train_losses.append(train_loss)
    all_valid_losses.append(valid_loss) 

    if epoch > 3:
        plt.plot(np.arange(3,epoch), np.array(all_train_losses[3:]), color="b", label="train MSE")
        plt.plot(np.arange(3,epoch), np.array(all_valid_losses[3:]), color="r", label="valid MSE")
        plt.title('Epoch %s: train MSE %s, validation MSE %s' % (epoch, train_loss, valid_loss))
        plt.xlabel("epochs")
        plt.ylabel("MSE")
        plt.legend()
        #plt.show()
        plt.savefig("plots/plot_24.png")
        plt.close()
    if valid_loss < best_val:
        torch.save({
                        'epoch': epoch,
                        'model_state_dict': interaction_network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'all_train_losses': all_train_losses,
                        'all_valid_losses' : all_valid_losses,
                    }, os.path.join("ckpts", "ckpt_new24_%d.p"%epoch))
