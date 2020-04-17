
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

from tqdm import tqdm
import matplotlib.pyplot as plt


# data = gen(n_objects, True)

data = np.load(os.path.join("data","featurized_train.npy"))

print("Data shape: ", data.shape) 
train_dataloader, validation_dataloader = get_dataloader(data, batch_size, USE_CUDA, object_dim, n_objects, relation_dim)

# Relation-centric Neural Network
# This NN takes all information about relations in the graph and outputs effects of all interactions between objects.



interaction_network = InteractionNetwork(n_objects, object_dim, n_relations, relation_dim, effect_dim)

if USE_CUDA:
    interaction_network = interaction_network.cuda()
    
optimizer = optim.Adam(interaction_network.parameters())
criterion = nn.MSELoss()


# Training

def validate(validation_loader, model, criterion):
    losses = []
    for dp, target in validation_loader:
        objects, sender_relations, receiver_relations, relation_info = dp
        predicted = interaction_network(objects, sender_relations, receiver_relations, relation_info)
        loss = criterion(predicted, target)  
        losses.append(loss.item())
    return np.mean(losses)



n_epoch = 200
batches_per_epoch = 100

all_train_losses = []
all_valid_losses = []

for epoch in tqdm(range(1,n_epoch+1)):
    losses = []
    for dp, target in train_dataloader:
        objects, sender_relations, receiver_relations, relation_info = dp
        predicted = interaction_network(objects, sender_relations, receiver_relations, relation_info)
        loss = criterion(predicted, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    train_mse = np.mean(losses)
    valid_mse = validate(validation_dataloader, interaction_network, criterion)  
    all_train_losses.append(train_mse)
    all_valid_losses.append(valid_mse) 

    if epoch > 3:
        plt.plot(np.arange(3,epoch), np.array(all_train_losses[3:]), color="b", label="train MSE")
        plt.plot(np.arange(3,epoch), np.array(all_valid_losses[3:]), color="r", label="valid MSE")
        plt.title('Epoch %s: train MSE %s, validation MSE %s' % (epoch, train_mse, valid_mse))
        plt.xlabel("epochs")
        plt.ylabel("MSE")
        plt.legend()
        #plt.show()
        plt.savefig("plots/plot.png")
        plt.close()
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': interaction_network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'all_train_losses': all_train_losses,
                    'all_valid_losses' : all_valid_losses,
                }, os.path.join("ckpts", "ckpt.p"))
