
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
#data = data[:10]
print("Data shape: ", data.shape) 
train_dataloader, validation_dataloader = get_dataloader(data, batch_size, USE_CUDA, object_dim, n_objects, relation_dim)



interaction_network = InteractionNetwork(n_objects, object_dim, n_relations, relation_dim, effect_dim)
device = "cpu"
if USE_CUDA:
    interaction_network = interaction_network.cuda()
    device= "cuda"
optimizer = optim.Adam(interaction_network.parameters())
#criterion = nn.MSELoss()


n_epoch = 200
batches_per_epoch = 100
loss_methods = {k: nn.MSELoss(reduction='none') for k in mse_indices}
loss_methods.update({k: nn.CrossEntropyLoss(reduction='none') for k in binary_indices})
loss_methods.update({k: nn.CrossEntropyLoss(reduction='none') for k in categorical_indices})


def add_to_epoch_loss_dict(epoch_loss_dict, loss_dict):
    for key in loss_dict:
        if key not in epoch_loss_dict:
            epoch_loss_dict[key] = []
        epoch_loss_dict[key].append(loss_dict[key])
    return epoch_loss_dict


def validate(validation_loader, model):
    losses = []
    epoch_val_loss_dict = dict()
    for dp, target in validation_loader:
        objects, sender_relations, receiver_relations, relation_info = dp
        predicted = interaction_network(objects, sender_relations, receiver_relations, relation_info)
        loss, loss_dict = calculate_total_loss(predicted, target)
        epoch_val_loss_dict = add_to_epoch_loss_dict(epoch_val_loss_dict, loss_dict)
        #loss = criterion(predicted, target)  
        losses.append(loss.item())
    return np.mean(losses), epoch_val_loss_dict


def mae(predicted, target):
    diff = torch.abs(predicted - target)
    diff = diff.reshape(-1, predicted.shape[-1])
    mae_mean = diff.mean(dim=0)
    return mae_mean

def add_vars(writer, diff, epoch):
    
    for key, val in index_mapping.items():
        if len(val)>1:
            continue # skip categorical vars
        writer.add_scalar(key, diff[val[0]], epoch)




def is_present_mask(is_present_target):
    
    is_present = is_present_target[:,0] == 1
    is_present == is_present.float()
    return is_present

def average_out_dict(my_dict):
    new_dict = dict()
    for key, val in my_dict.items():
        new_dict[key] = np.mean(val)
    return new_dict


def calculate_total_loss(predicted, target):
    """returns batch loss and batch loss dict"""

    loss_dict = dict()
    loss = 0
    mask = None

    for key, val in index_mapping.items():
        if key == "is_present":
            cur_target = target[:,:,val[0]:val[-1]+1].reshape(-1,len(val))
            mask = is_present_mask(cur_target)
            break

    denominator = mask.sum()
    for key, val in index_mapping.items():
        cur_criterion = loss_methods[key]
        if key == "is_present":
            print("skipping: ", key)
            continue

        cur_target = target[:,:,val[0]:val[-1]+1].reshape(-1,len(val))
        
        if "CrossEntropyLoss" in str(cur_criterion):
            _, cur_target = cur_target.max(dim=1)
            cur_target = cur_target.long()

        cur_predicted = predicted[:,:,val[0]:val[-1]+1].reshape(-1, len(val))
        cur_loss = cur_criterion(cur_predicted, cur_target).reshape(-1)

        assert cur_loss.shape == mask.shape
        cur_loss = (mask * cur_loss).sum()
        
        if denominator > 0:
            cur_loss /= denominator

        loss_dict[key] = cur_loss.item()
        loss +=  cur_loss
    return loss, loss_dict




all_train_losses = []
all_valid_losses = []
best_val = 1000000000
for epoch in tqdm(range(1,n_epoch+1)):
    losses = []
    epoch_train_loss_dict = dict()
    running_mae = torch.zeros(data.shape[-1]).to(device)
    for dp, target in train_dataloader:
        objects, sender_relations, receiver_relations, relation_info = dp
        predicted = interaction_network(objects, sender_relations, receiver_relations, relation_info)
        mae_mean = mae(predicted, target)
        running_mae += mae_mean
        
        #loss = criterion(predicted, target)
        loss, loss_dict = calculate_total_loss(predicted, target)
        epoch_train_loss_dict = add_to_epoch_loss_dict(epoch_train_loss_dict, loss_dict)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    running_mae /= len(train_dataloader)
    add_vars(writer, running_mae, epoch)
    train_loss = np.mean(losses)
    valid_loss, epoch_val_loss_dict = validate(validation_dataloader, interaction_network) 
    epoch_train_loss_dict = average_out_dict(epoch_train_loss_dict)
    epoch_val_loss_dict = average_out_dict(epoch_val_loss_dict)

    print("valid_loss: ", valid_loss)
    writer.add_scalars("Losses", {'train_loss': train_loss, 'valid_loss' : valid_loss}, epoch)
    for key, _ in epoch_train_loss_dict.items():
        writer.add_scalars("Per var loss: %s"%key, {'train_loss': epoch_train_loss_dict[key], \
            'valid_loss' : epoch_val_loss_dict[key]}, epoch)
    
    all_train_losses.append(train_loss)
    all_valid_losses.append(valid_loss) 

    if epoch > 3:
        plt.plot(np.arange(3,epoch), np.array(all_train_losses[3:]), color="b", label="train loss")
        plt.plot(np.arange(3,epoch), np.array(all_valid_losses[3:]), color="r", label="valid loss")
        plt.title('Epoch %s: train loss %s, validation loss %s' % (epoch, train_loss, valid_loss))
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        #plt.show()
        plt.savefig("plots/plot_24.png")
        plt.close()
    if valid_loss < best_val:
        print("Saving ckpt epoch ", epoch)
        torch.save({
                        'epoch': epoch,
                        'model_state_dict': interaction_network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'all_train_losses': all_train_losses,
                        'all_valid_losses' : all_valid_losses,
                    }, os.path.join("ckpts", "ckpt_new24_%d.p"%epoch))
