
import os
import random
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()
        
        self.output_size = output_size
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        batch_size, n_relations, input_size = x.size()
        x = x.view(-1, input_size)
        x = self.layers(x)
        x = x.view(batch_size, n_relations, self.output_size)
        return x


# Object-centric Neural Network
# This NN takes information about all objects and effects on them, then outputs prediction of the next state of the graph.</p>


class ObjectModel(nn.Module):
    def __init__(self, input_size, hidden_size, object_dim):
        super(ObjectModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
           # nn.Linear(hidden_size, 2), #speedX and speedY
           nn.Linear(hidden_size, object_dim)
        )
        
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_objects, input_size]
        Returns:
            [batch_size * n_objects, object_dim] 
        '''
        input_size = x.size(2)
        x = x.view(-1, input_size)
        return self.layers(x)


# Interaction Network
# IN involves only matrix operations that do not contain learnable parameters.


class InteractionNetwork(nn.Module):
    def __init__(self, n_objects, object_dim, n_relations, relation_dim, effect_dim):
        super(InteractionNetwork, self).__init__()

        self.relational_model = RelationalModel(2*object_dim + relation_dim, effect_dim, 150)
        self.object_model     = ObjectModel(object_dim + effect_dim, 100, object_dim)
        self.n_objects = n_objects
        self.object_dim = object_dim
    
    def forward(self, objects, sender_relations, receiver_relations, relation_info):
        if len(sender_relations.shape) ==2 :
            sender_relations.unsqueeze_(0)
        if len(receiver_relations.shape)==2 :
            receiver_relations.unsqueeze_(0)
        if len(relation_info.shape) == 2:
            relation_info.unsqueeze_(0)
        senders   = sender_relations.permute(0, 2, 1).bmm(objects)
        receivers = receiver_relations.permute(0, 2, 1).bmm(objects)
        effects = self.relational_model(torch.cat([senders, receivers, relation_info], 2))
        effect_receivers = receiver_relations.bmm(effects)
#        predicted = self.object_model(torch.cat([objects, effect_receivers], 2))
       
        predicted = self.object_model(torch.cat([objects, effect_receivers], 2)).reshape(-1, self.n_objects, self.object_dim)


        return predicted

