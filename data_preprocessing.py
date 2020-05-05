import json
import os
from tqdm import tqdm
import numpy as np
from random import randint
from matplotlib import image
from hyperparams import SHAPE_2_IDX, IDX_2_SHAPE
from utils import merge_dicts
from transformations import world2camera_coord, world2camera_rotation

# the list indicates which indices in the feature vector pertain to the key
binary_indices = {"is_present":[0,1], "is_occluder":[2,3], "is_visible":[4,5]}
mse_indices =     {
                    "location_x":[6], "location_y":[7], "location_z":[8],
                    "velocity_x":[9], "velocity_y":[10], "velocity_z":[11],
                    "scale_x":[12], "scale_y":[13], "scale_z":[14],
                    "rotation_roll":[15], "rotation_pitch":[16], "rotation_yaw":[17],
                    "speed":[18], "friction":[19], "mass":[20]
                    }

categorical_indices = {"shape":[21,22,23,24]}
index_mapping = merge_dicts(binary_indices, mse_indices, categorical_indices)

flatten = lambda lis: [item for sublist in lis for item in sublist]


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

def get_all_shapes(data_path, path="data/all_shapes.txt"):
	"""Get all possible shapes of objects"""

	if os.path.exists(path):
		return set(open(path, 'r').read().strip().split(","))

	pbar = tqdm(total=len(list(os.listdir(data_path))))
	shapes = set()

	for datafolder in os.listdir(data_path):
		json_path = os.path.join(data_path, datafolder, 'status.json')
		frames = json.load(open(json_path, 'r'))['frames']

		for frame in frames:
			for obj_name in frame:
				if "shape" in frame[obj_name]:
					shapes.add(frame[obj_name]["shape"])
		    
		pbar.update()

	with open(path, 'w') as fp:
		fp.write(", ".join(shapes))

	return shapes


def transform_features(feat, camera):
	"""Transform x,y,z,yaw,pitch,roll coordinates for feature vector"""
	p_w = [feat[mse_indices["location_x"][0]], feat[mse_indices["location_y"][0]], feat[mse_indices["location_z"][0]]]
	new_coord = world2camera_coord(p_w, camera)
	feat[mse_indices["location_x"][0]] = new_coord[0]
	feat[mse_indices["location_y"][0]] = new_coord[1]
	feat[mse_indices["location_z"][0]] = new_coord[2]

	obj_rot = [feat[mse_indices["rotation_pitch"][0]], feat[mse_indices["rotation_yaw"][0]], feat[mse_indices["rotation_roll"][0]]]
	new_rot = world2camera_rotation(obj_rot, camera)
	feat[mse_indices["rotation_pitch"][0]] = new_rot[0]
	feat[mse_indices["rotation_yaw"][0]] = new_rot[1]
	feat[mse_indices["rotation_roll"][0]] = new_rot[2] 
	return feat


def featurize(obj_dict, is_occluder, is_visible, shape, camera):
	"""Returns a list of features for object"""
	result = [0 for _ in range(max(flatten(index_mapping.values()))+1)]

	# binary variables are represented as one-hot encoding
	for key, val in binary_indices.items():
		if key == "is_present":
			result[val[0]] = 1 # obj is definitely present
			result[val[1]] = 0
		elif key == "is_occluder":
			result[val[0]] = 1 if is_occluder else 0
			result[val[1]] = 0 if is_occluder else 1
		elif key == "is_visible":
			result[val[0]] = 1 if is_visible else 0
			result[val[1]] = 0 if is_visible else 1
		else:
			raise ValueError("You're wrong")


	# encode the continious variables
	for key, idx_value in mse_indices.items():
		assert len(idx_value) == 1
		idx_value = idx_value[0]
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

	# encode the categorical variables (shape) as one-hot
	for key, val in categorical_indices.items():
		if key == "shape":
			idx_rel = SHAPE_2_IDX[shape]
			result[val[idx_rel]] = 1
		else:
			raise ValueError("You have a mistake")

	transformed_result = transform_features(result, camera)
	return transformed_result


def get_visible_bit(video_path, frame_idx, mask_id):
	"""Returns a value 1 or 0 depending on whether object is visible or not"""
	img = image.imread(os.path.join(video_path, "masks", "masks_{}.png".format(str(frame_idx+1).zfill(3))))
	mask_id = mask_id/255
	mask = img == mask_id
	mask[mask > 0] = 1
	overlap = np.count_nonzero(mask)
	if overlap >= 25: # if area greater than 25, object is visible
		return 1
	else:
		return 0


def get_free_index(objname2index):
	"""Get free index in the obj_name:idx map to assign to object and track it across frames"""
	values = set(objname2index.values())
	possible_indices = set(range(5))
	free = [pos for pos in possible_indices if pos not in values]
	return free[0]


def process_video(video_path, max_obj=5):
	"""Returns a matrix of dim num_frames x max_obj x num_features encoding the video"""
	json_path = os.path.join(video_path, 'status.json')
	video_status = json.load(open(json_path, 'r'))
	frames = video_status['frames']
	camera = video_status['header']['camera']

	video_info = []
	objname2index = dict() # object to idx in obj
	for frame_idx, frame in enumerate(frames):
		# initialize to empty features for all objects
		frame_info = [[0 for _ in range(max(flatten(index_mapping.values()))+1)] for _ in range(max_obj)]
		for obj_fr in frame_info:
			obj_fr[1] = 1
		for _, obj_name in enumerate(frame):
			if obj_name != "masks":
				is_occluder = "occluder" in obj_name.lower()
				if is_occluder:
					frame[obj_name]["shape"] = "Occluder"
				if obj_name not in objname2index:
					# each unique object gets an index indictaing its place among the max_obj
					index = get_free_index(objname2index)
					objname2index[obj_name] = index

				if obj_name in frame["masks"]:
					is_visible = get_visible_bit(video_path, frame_idx, frame["masks"][obj_name])
				else:
					is_visible = 0
				# featurize the object
				obj_info = featurize(frame[obj_name], is_occluder, is_visible, frame[obj_name]["shape"], camera)
				# add the object to its index position
				frame_info[objname2index[obj_name]] = obj_info

		video_info.append(frame_info)
	return video_info


def run_processing(data_folder, max_obj, outfile):
	"""Runs the video processing pipeline for data_folder. 
	Final result has shape: num_videos x num_frames x max_obj x num_features =  (15000, 100, 5, 25) in our case"""
	print("Starting processing for data in {}...".format(data_folder))
	final_result = []
	pbar = tqdm(total=len(list(os.listdir(data_folder))))
	for video_id in os.listdir(data_folder):
		video_path = os.path.join(data_folder, video_id)
		video_info = process_video(video_path, max_obj)
		final_result.append(video_info)
		pbar.update()
	#print(final_result[0])
	final_np = np.array(final_result)
	print("Final result has shape: {}".format(final_np.shape))
	print("Sample video representation: {}".format(final_np[randint(0, final_np.shape[0])]))
	np.save(outfile, final_np)
	print("Processing complete. Saved results in {}".format(outfile))


if __name__ == "__main__":
	data_folder = "../intphys/train"
	outfile = "data/featurized_train_24.npy"
	max_obj = get_max_num_obj(data_folder)
	#print(get_all_shapes(data_folder))
	run_processing(data_folder, max_obj, outfile)
	
