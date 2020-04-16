
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

USE_CUDA = True



from tqdm import tqdm
import matplotlib.pyplot as plt


n_objects  = 5 # number of planets(nodes)
object_dim = 17 

n_relations  = n_objects * (n_objects - 1) # number of edges in fully connected graph
relation_dim = 1

effect_dim = 100 #effect's vector size

batch_size = 1024

# data = gen(n_objects, True)

data = np.load(os.path.join("data","featurized_train.npy"))

print("Data shape: ", data.shape) 
train_dataloader, validation_dataloader = get_dataloader(data, batch_size, USE_CUDA, object_dim, n_objects, n_relations, relation_dim)


# def get_batch(data, batch_size):
#     rand_idx  = [random.randint(0, len(data) - 2) for _ in range(batch_size)]
#     label_idx = [idx + 1 for idx in rand_idx]
    
#     batch_data = data[rand_idx] 
#     label_data = data[label_idx]
    
#     #objects = batch_data[:,:,:5]
#     objects = batch_data
    
    
#     #receiver_relations, sender_relations - onehot encoding matrices
#     #each column indicates the receiver and sender object’s index
    
#     receiver_relations = np.zeros((batch_size, n_objects, n_relations), dtype=float);
#     sender_relations   = np.zeros((batch_size, n_objects, n_relations), dtype=float);
    
#     cnt = 0
#     for i in range(n_objects):
#         for j in range(n_objects):
#             if(i != j):
#                 receiver_relations[:, i, cnt] = 1.0
#                 sender_relations[:, j, cnt]   = 1.0
#                 cnt += 1
    
#     #There is no relation info in solar system task, just fill with zeros
#     relation_info = np.zeros((batch_size, n_relations, relation_dim))
#    # target = label_data[:,:,3:]
#     target = label_data
    
#     objects            = Variable(torch.FloatTensor(objects))
#     sender_relations   = Variable(torch.FloatTensor(sender_relations))
#     receiver_relations = Variable(torch.FloatTensor(receiver_relations))
#     relation_info      = Variable(torch.FloatTensor(relation_info))
# #    target             = Variable(torch.FloatTensor(target)).view(-1, 2)
#     target             = Variable(torch.FloatTensor(target)).view(-1, object_dim)

                       
#     if USE_CUDA:
#         objects            = objects.cuda()
#         sender_relations   = sender_relations.cuda()
#         receiver_relations = receiver_relations.cuda()
#         relation_info      = relation_info.cuda()
#         target             = target.cuda()
    
#     return objects, sender_relations, receiver_relations, relation_info, target


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



n_epoch = 100
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
