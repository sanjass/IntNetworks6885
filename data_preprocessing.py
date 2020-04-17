import json
import os
from tqdm import tqdm
import numpy as np
from random import randint


def get_max_num_obj(data_path, path="data/max_objects.txt"):
	"""Get the max number of objects in a set of videos"""

	if os.path.exists(path):
		return int(open(path, 'r').read().strip())

	pbar = tqdm(total=len(list(os.listdir(data_path))))
	max_overall = 0

	for datafolder in os.listdir(data_path):
		json_path = os.path.join(data_path, datafolder, 'status.json')
		frames = json.load(open(json_path, 'r'))['frames']
		max_for_video = 0
		# subtract 1 bc one entry is "masks" which is not an object
		for frame in frames:
		    max_for_video = max(max_for_video, len(frame)-1) 
		    print(max_for_video)
		max_overall = max(max_for_video, max_overall)
		pbar.update()

	with open(path, 'w') as fp:
		fp.write(str(max_overall))

	return max_overall

# FORMAT for features of an object:
# [present_or_not (1 or 0),  - 1
# type of obj (occluder=1, non-occluder 0) -1
# location (x,y,z), - 3
# velocity (x,y,x), -3,
# scale (x,y,z), -3
# rotation (roll, pitch, yaw) -3,
# speed (only for occluders), -1
# friction, -1,
# mass (only for obj), -1] 

index_mapping = {"location_x":2, "location_y":3, "location_z":4, "velocity_x":5, "velocity_y":6, "velocity_z":7, "scale_x":8, "scale_y":9, "scale_z":10,
                 "rotation_roll":11, "rotation_pitch":12, "rotation_yaw":13, "speed":14, "friction":15, "mass":16}

def featurize(obj_dict, is_occluder):
	"""Returns a list of features for object"""
	result = [0 for _ in range(max(index_mapping.values())+1)]
	result[0] = 1 # obj is present

	result[1] = 1 if is_occluder else 0

	for key, idx_value in index_mapping.items():
		if "_" in key:
			subkeys = key.split("_")
			sub1 = subkeys[0]
			sub2 = subkeys[1]
			if sub1 not in obj_dict:
				result[idx_value] = 0 # put 0 if info not present for occluders
			else:
				result[idx_value] = obj_dict[sub1][sub2] # populate the result encoding
		else:
			 if key not in obj_dict:
			 	result[idx_value] = 0
			 else:
			 	result[idx_value] = obj_dict[key]
	return result



def get_free_index(objname2index):
	"""Get free index in the obj_name:idx map"""
	values = set(objname2index.values())
	possible_indices = set(range(5))
	free = [pos for pos in possible_indices if pos not in values]
	return free[0]


def process_video(video_path, max_obj=5):
	"""Returns a matrix of dim num_frames x max_obj x num_features"""
	json_path = os.path.join(video_path, 'status.json')
	frames = json.load(open(json_path, 'r'))['frames']
	video_info = []
	objname2index = dict() # object to idx in obj
	for frame in frames:
		frame_info = [[0 for _ in range(max(index_mapping.values())+1)] for _ in range(max_obj)]
		for _, obj_name in enumerate(frame):
			if obj_name != "masks":
				is_occluder = "occluder" in obj_name
				if obj_name not in objname2index:
					index = get_free_index(objname2index)
					objname2index[obj_name] = index

				obj_info = featurize(frame[obj_name], is_occluder)
				frame_info[objname2index[obj_name]] = obj_info
		video_info.append(frame_info)
	return video_info


def run_processing(data_folder, max_obj, outfile):
	"""Runs the video processing pipeline for data_folder. 
	Final result has shape: num_videos x num_frames x max_obj x num_features =  (15000, 100, 5, 17) in our case"""
	print("Starting processing for data in {}...".format(data_folder))
	final_result = []
	pbar = tqdm(total=len(list(os.listdir(data_folder))))
	for video_id in os.listdir(data_folder):
		video_path = os.path.join(data_folder, video_id)
		video_info = process_video(video_path, max_obj)
		final_result.append(video_info)
		pbar.update()
	final_np = np.array(final_result)
	print("Final result has shape: {}".format(final_np.shape))
	print("Sample video representation: {}".format(final_np[randint(0, final_np.shape[0])]))
	np.save(outfile, final_np)
	print("Processing complete. Saved results in {}".format(outfile))


if __name__ == "__main__":
	data_folder = "../intphys/train"
	outfile = "data/train.npy"
	max_obj = get_max_num_obj(data_folder)
	run_processing(data_folder, max_obj, outfile)
	
