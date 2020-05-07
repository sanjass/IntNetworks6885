from MyDataset import MyDataset
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from hyperparams import *
from models import InteractionNetwork 
import skvideo.io
from multiprocessing import Process, cpu_count
from multiprocessing.pool import ThreadPool
import glob
import os
import numpy as np
from PIL import Image


def build_recursive_case_paths(input_folder, cases):
    if "scene" not in os.listdir(input_folder):
        to_recurse = sorted(list(os.path.join(input_folder, sub_folder) for sub_folder in os.listdir(input_folder)))
        for new_folder in to_recurse:
            if os.path.isdir(new_folder):
                build_recursive_case_paths(new_folder, cases)
    else:
        cases.append(input_folder)
    return cases


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
    if shuffle_dataset:
    	train_sampler = SubsetRandomSampler(train_indices)
    	valid_sampler = SubsetRandomSampler(val_indices)
    else:
    	train_sampler = SequentialSampler(train_indices)
    	valid_sampler = SequentialSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader



def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def make_video(save_path, ims, fps=8, duration=None):
    """ 
    Creates a video given an array of images. Uses FFMPEG backend.
    Depending on the FFMPEG codec supported you might have to change pix_fmt
    To change the quality of the saved videos, change -b (the encoding bitrate)
    Args:
      save_path: path to save the video
      ims: an array of images
      fps: frames per seconds to save the video
      duration: the duration of the video, if not None, will override fps.
    > ims = [im1, im2, im3]
    > make_video('video.mp4', ims, fps=10)
    """
    print("Started making mp4 summary video")
    if duration is not None:
        fps = len(ims) / duration
    skvideo.io.vwrite(save_path, ims,
                      inputdict={'-r': str(fps)},
                      outputdict={'-r': str(fps), 
                                  '-pix_fmt': 'yuv420p',
                                   '-vcodec': 'libx264', 
                                  '-b': '10000000'}, verbosity=1)
    print("Finished making mp4 summary video")
    return






def save_video(save_path, images_path):
    images_files = sorted(glob.glob(os.path.join(images_path, "*.png")))
    with ThreadPool(cpu_count() * 4) as p:
        images = p.map(lambda x: np.array(Image.open(x)), images_files)

    make_video(save_path, images, duration=10)
