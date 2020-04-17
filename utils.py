from MyDataset import MyDataset
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from hyperparams import *
from models import InteractionNetwork 



def load_model(ckpt_path):
	model = InteractionNetwork(n_objects, object_dim, n_relations, relation_dim, effect_dim)
	model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
	model.eval()
	return model


def get_dataloader(data, batch_size, USE_CUDA=True, object_dim=100, n_objects=5, relation_dim=1, validation_split=0.3,\
    random_seed=42, shuffle_dataset = True):

	n_relations = n_objects * (n_objects - 1)
	dataset = MyDataset(data, USE_CUDA, object_dim, n_objects, n_relations, relation_dim)
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	

	split = int(np.floor(validation_split * dataset_size))
	if shuffle_dataset :
	    np.random.seed(random_seed)
	    np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]

	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)

	train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
	                                           sampler=train_sampler)
	validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
	                                                sampler=valid_sampler)
	return train_loader, validation_loader
