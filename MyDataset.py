from torch.utils.data.dataset import Dataset
import numpy as np
from torch.autograd import Variable
import torch


class MyDataset(Dataset):
    def __init__(self, raw_data, USE_CUDA, object_dim, n_objects, n_relations, relation_dim):
        self.raw_data = raw_data
        # the dataset is supposed to be stored in a list of pairs [input, label]
        self.data = self.concatenate_across_videos(raw_data)
        self.object_dim = object_dim
        self.n_objects = n_objects
        self.USE_CUDA = USE_CUDA
        self.n_relations = n_relations
        self.relation_dim = relation_dim

    def concatenate_across_videos(self, raw_data):
        datapoints = []
        for video in raw_data:
            frame_indices = np.array(list(range(len(video)-1)))
            objects = video[frame_indices]
            label_idx = frame_indices + 1
            targets = video[label_idx]
            for idx, _ in enumerate(objects):
                datapoints.append([objects[idx], targets[idx]])
        return datapoints


    def __getitem__(self, index):

        objects, target = self.data[index]
        
        #receiver_relations, sender_relations - onehot encoding matrices
        #each column indicates the receiver and sender object’s index
        
        receiver_relations = np.zeros((self.n_objects, self.n_relations), dtype=float);
        sender_relations   = np.zeros((self.n_objects, self.n_relations), dtype=float);
        
        cnt = 0
        for i in range(self.n_objects):
            for j in range(self.n_objects):
                if(i != j):
                    receiver_relations[i, cnt] = 1.0
                    sender_relations[j, cnt]   = 1.0
                    cnt += 1
        
        #There is no relation info in solar system task, just fill with zeros
        relation_info = np.zeros((self.n_relations, self.relation_dim))
       # target = label_data[:,:,3:]
        
        objects            = Variable(torch.FloatTensor(objects))
        sender_relations   = Variable(torch.FloatTensor(sender_relations))
        receiver_relations = Variable(torch.FloatTensor(receiver_relations))
        relation_info      = Variable(torch.FloatTensor(relation_info))
    #    target             = Variable(torch.FloatTensor(target)).view(-1, 2)
        target             = Variable(torch.FloatTensor(target)).view(-1, self.object_dim)

                           
        if self.USE_CUDA:
            objects            = objects.cuda()
            sender_relations   = sender_relations.cuda()
            receiver_relations = receiver_relations.cuda()
            relation_info      = relation_info.cuda()
            target             = target.cuda()
        
        return [objects, sender_relations, receiver_relations, relation_info], target


    def __len__(self):
        return len(self.data)