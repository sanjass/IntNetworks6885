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
            objects_all_frames = video[frame_indices]
            label_idx = frame_indices + 1
            targets = video[label_idx]
            for idx, _ in enumerate(objects_all_frames):
                datapoints.append([objects_all_frames[idx], targets[idx]])
        return datapoints

    @staticmethod
    def construct_input(objects, target=None, n_objects=5, n_relations=20, relation_dim=1, USE_CUDA=True):
        receiver_relations = np.zeros((n_objects, n_relations), dtype=float);
        sender_relations   = np.zeros((n_objects, n_relations), dtype=float);
        
        cnt = 0
        for i in range(n_objects):
            for j in range(n_objects):
                if(i != j):
                    receiver_relations[i, cnt] = 1.0
                    sender_relations[j, cnt]   = 1.0
                    cnt += 1
        #There is no relation info in solar system task, just fill with zeros
        relation_info = np.zeros((n_relations, relation_dim))
        objects            = Variable(torch.FloatTensor(objects))
        sender_relations   = Variable(torch.FloatTensor(sender_relations))
        receiver_relations = Variable(torch.FloatTensor(receiver_relations))
        relation_info      = Variable(torch.FloatTensor(relation_info))
        if target is not None:
            target         = Variable(torch.FloatTensor(target))#.view(-1, self.object_dim)

        if USE_CUDA:
            objects            = objects.cuda()
            sender_relations   = sender_relations.cuda()
            receiver_relations = receiver_relations.cuda()
            relation_info      = relation_info.cuda()
            if target is not None:
                target         = target.cuda()

        return [objects, sender_relations, receiver_relations, relation_info], target

    def __getitem__(self, index):

        objects, target = self.data[index]
        
        #receiver_relations, sender_relations - onehot encoding matrices
        #each column indicates the receiver and sender object’s index
        
        return self.construct_input(objects, target, self.n_objects, self.n_relations, self.relation_dim, self.USE_CUDA)


    def __len__(self):
        return len(self.data)