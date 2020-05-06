import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from hyperparams import *
from utils import load_model
import numpy as np
from itertools import product, combinations
from MyDataset import MyDataset
from data_preprocessing import process_video, mse_indices
from utils import get_dataloader
import os
from tqdm import tqdm
from utils import save_video
import matplotlib.image as mpimg
import torch.nn.functional as F
from transformations import rotate_vector



ckpt_to_use = "ckpts/ckpt_new24_91.p"

def plot_object(ax, center_x, center_y, center_z, rot_pitch, rot_yaw, rot_roll, radius, color, title):

	alpha = 0.3
	u = np.linspace(0, 2 * np.pi, 50)
	v = np.linspace(0, np.pi, 50)
	limit = 1200

	x = radius * np.outer(np.cos(u), np.sin(v)) + center_x
	y = radius * np.outer(np.sin(u), np.sin(v)) + center_y
	z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center_z

	sphere = ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=30)
	ax.set_xlim(-limit,limit)
	ax.set_ylim(-limit,limit)
	ax.set_zlim(-limit,limit)
	if "ground" in title:
		ax.set_title(title + "\nloc: x={}, y={}, z={}".format(round(center_x,2), round(center_y,2), round(center_z,2)))
	else:
		ax.set_title(title)
	r = [-300, 300]
	for s, e in combinations(np.array(list(product(r,r,r))), 2):
		#print(s, e)
		if np.sum(np.abs(s-e)) == r[1]-r[0]:
			s = rotate_vector(s, rot_yaw, rot_pitch, rot_roll)
			e = rotate_vector(e, rot_yaw, rot_pitch, rot_roll)
			result = zip(*(s,e))
			xs, ys, zs = result
			xs = [x+center_x for x in xs]
			ys = [y+center_y for y in ys]
			zs = [z+center_z for z in zs]
			ax.plot3D(xs, ys, zs, color="b")

	#ax.plot_surface(xdata, ydata, zdata, rstride=3, cstride=3, color='b', shade=0.5)


def plot_scene(img_path, ax):
	my_img = mpimg.imread(img_path)
	ax.imshow(my_img)
	ax.set_title("ground truth")

def get_predictions(initial_frame_objects, model=None):
	if model is None:
		model = load_model(ckpt_to_use)

	objects = initial_frame_objects
	predictions = [objects]

	for i in range(1, 100):
		dp, target = MyDataset.construct_input(objects, USE_CUDA=False)
		objects, sender_relations, receiver_relations, relation_info = dp
		output = model(objects, sender_relations, receiver_relations, relation_info)
		predictions.append(output)
		objects = output

	return predictions


def analyze_video(video_path, use_ground_truth=False):
	model = load_model(ckpt_to_use)
	video_info = process_video(video_path)
	data = np.array([video_info])
	#print([data[0][i][0][2] for i in range(data.shape[1])])
	dataloader, _ = get_dataloader(data, 1, USE_CUDA, object_dim, n_objects, relation_dim, validation_split=0, shuffle_dataset=False)
	ground_truths = []

	if use_ground_truth:
		print("Using ground truth at time t to predict t+1")

	if use_ground_truth:
		predictions = []
		for idx, tup in enumerate(dataloader):
			dp, _ = tup
			objects, _, _, _ = dp

			dp, target = MyDataset.construct_input(objects.cpu(), USE_CUDA=False)
			objects, sender_relations, receiver_relations, relation_info = dp

			ground_truths.append(objects)
			if idx == 0:
				predictions.append(objects)
			if idx != len(dataloader)-1:
				output = model(objects, sender_relations, receiver_relations, relation_info)
				predictions.append(output)

	else:
		for idx, tup in enumerate(dataloader):
			dp, _ = tup
			objects, _, _, _ = dp
			if idx == 0:
				predictions = get_predictions(objects.cpu(), model)
			ground_truths.append(objects)

	assert len(ground_truths) == len(predictions)

	return ground_truths, predictions


def plot_video(video_path, ground_truths, predictions, save_dir="videos"):
	

	#fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7, 3.5))
	colors = ['r','b','g','y','m']

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	
	scenes = list(os.listdir(os.path.join(video_path, "scene")))
	scenes.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))


	for frame_idx in tqdm(range(len(ground_truths))):
		fig = plt.figure(figsize=(15, 4.0))
		ax1 = fig.add_subplot(131)
		ax2 = fig.add_subplot(132, projection='3d')
		ax3 = fig.add_subplot(133, projection='3d')

		plot_scene(os.path.join(video_path, "scene", scenes[frame_idx]), ax1)

		ground_truth_objects = ground_truths[frame_idx][0]
		predicted_objects = predictions[frame_idx][0]
		radius = 200
		
		for obj_idx in range(ground_truth_objects.shape[0]):
			obj_gr = ground_truth_objects[obj_idx]
			obj_pr = predicted_objects[obj_idx]

			if obj_gr[0].item() == 1: # object is present
				loc_x, loc_y, loc_z = obj_gr[mse_indices["location_x"]].item(), obj_gr[mse_indices["location_y"]].item(), obj_gr[mse_indices["location_z"]].item()
				rot_pitch, rot_yaw, rot_roll = obj_gr[mse_indices["rotation_pitch"]].item(), obj_gr[mse_indices["rotation_yaw"]].item(), obj_gr[mse_indices["rotation_roll"]].item()
				plot_object(ax2, loc_x, loc_y, loc_z, rot_pitch, rot_yaw, rot_roll, radius*(obj_idx+1)*0.5, colors[obj_idx], "ground truth")

			presence_log = F.log_softmax(obj_pr[:2], dim=0) 
			presence_bit = 1 if presence_log[0] > presence_log[1] else 0
			print("gr: ", obj_gr[0].item(), " pr: ",presence_bit)
			if presence_bit == 1: # object is present
			#if True:
				loc_x, loc_y, loc_z = obj_pr[mse_indices["location_x"]].item(), obj_pr[mse_indices["location_y"]].item(), obj_pr[mse_indices["location_z"]].item()
				rot_pitch, rot_yaw, rot_roll = obj_pr[mse_indices["rotation_pitch"]].item(), obj_pr[mse_indices["rotation_yaw"]].item(), obj_pr[mse_indices["rotation_roll"]].item()
				plot_object(ax3, loc_x, loc_y, loc_z, rot_pitch, rot_yaw, rot_roll, radius*(obj_idx+1)*0.5, colors[obj_idx], "predicted")

		plt.savefig(os.path.join(save_dir, 'frame_{:02}.png'.format(frame_idx)))
		plt.close()




if __name__ == "__main__":
	print("Using ckpts: ", ckpt_to_use)
	video_path = "../intphys/train/00001"
	ground_truths, predictions = analyze_video(video_path, use_ground_truth=True)
	# print("ground truths: ",ground_truths)
	# print("predictions : ", predictions)

	plot_video(video_path, ground_truths, predictions)
	save_video("sample/{}.mp4".format(video_path.split("/")[-1]), "videos")
	