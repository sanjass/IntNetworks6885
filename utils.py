from MyDataset import MyDataset
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def get_dataloader(data, batch_size, USE_CUDA, object_dim, n_objects, n_relations, relation_dim, validation_split=0.3, random_seed=42):
	dataset = MyDataset(data, USE_CUDA, object_dim, n_objects, n_relations, relation_dim)
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	shuffle_dataset = True

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
