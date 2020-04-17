import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from hyperparams import *
from utils import load_model
from MyDataset import MyDataset
from data_preprocessing import process_video
from utils import get_dataloader
import os
from tqdm import tqdm
from utils import save_video
import numpy as np


def plot_object(ax, center_x, center_y, center_z, radius, color, title):

	alpha = 0.3
	u = np.linspace(0, 2 * np.pi, 50)
	v = np.linspace(0, np.pi, 50)
	limit = 2000

	x = radius * np.outer(np.cos(u), np.sin(v)) + center_x
	y = radius * np.outer(np.sin(u), np.sin(v)) + center_y
	z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center_z

	sphere = ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=30)
	ax.set_xlim(-limit,limit)
	ax.set_ylim(-limit,limit)
	ax.set_zlim(-limit,limit)
	ax.set_title(title)

	#ax.plot_surface(xdata, ydata, zdata, rstride=3, cstride=3, color='b', shade=0.5)



def get_predictions(initial_frame_objects, model=None):
	if model is None:
		model = load_model("ckpts/ckpts_71.p")

	objects = initial_frame_objects
	predictions = [objects]

	for i in range(1, 100):
		dp, target = MyDataset.construct_input(objects, USE_CUDA=False)
		objects, sender_relations, receiver_relations, relation_info = dp
		output = model(objects, sender_relations, receiver_relations, relation_info)
		predictions.append(output)
		objects = output

	return predictions


def analyze_video(video_path):
	model = load_model("ckpts/ckpts_71.p")
	video_info = process_video(video_path)
	data = np.array([video_info])
	#print([data[0][i][0][2] for i in range(data.shape[1])])
	dataloader, _ = get_dataloader(data, 1, USE_CUDA, object_dim, n_objects, relation_dim, validation_split=0, shuffle_dataset=False)
	ground_truths = []

	for idx, tup in enumerate(dataloader):
		dp, _ = tup
		objects, _, _, _ = dp
		if idx == 0:
			predictions = get_predictions(objects.cpu(), model)

		ground_truths.append(objects)

	return ground_truths, predictions


def plot_video(ground_truths, predictions, save_dir="videos"):
	

	#fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7, 3.5))
	colors = ['r','b','g','y','m']

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	for frame_idx in tqdm(range(len(ground_truths))):
		fig = plt.figure(figsize=(10, 3.5))
		ax1 = fig.add_subplot(121, projection='3d')
		ax2 = fig.add_subplot(122, projection='3d')
		ground_truth_objects = ground_truths[frame_idx].squeeze_(0)
		predicted_objects = predictions[frame_idx][0].squeeze_(0)
		radius = 200
		
		for obj_idx in range(ground_truth_objects.shape[0]):
			obj_gr = ground_truth_objects[obj_idx]
			obj_pr = predicted_objects[obj_idx]

			if obj_gr[0] == 1: # object is present
				loc_x, loc_y, loc_z = obj_gr[2].item(), obj_gr[3].item(), obj_gr[4].item()
				plot_object(ax1, loc_x, loc_y, loc_z, radius*(obj_idx+1), colors[obj_idx], "ground truth")


			if obj_pr[0] >= 0.5: # object is present
			#if True:
				loc_x, loc_y, loc_z = obj_pr[2].item(), obj_pr[3].item(), obj_pr[4].item()
				plot_object(ax2, loc_x, loc_y, loc_z, radius*(obj_idx+1), colors[obj_idx], "predicted")

		plt.savefig(os.path.join(save_dir, 'frame_{:02}.png'.format(frame_idx)))
		plt.close()




if __name__ == "__main__":
	video_path = "../intphys/train/00001"
	ground_truths, predictions = analyze_video(video_path)
	# print("ground truths: ",ground_truths)
	# print("predictions : ", predictions)
	plot_video(ground_truths, predictions)
	save_video("sample/{}.mp4".format(video_path.split("/")[-1]), "videos")
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# plot_object(ax,2,3,4, 3,'b')
	# plot_object(ax, 6,5,4, 5,'r') 

	# plt.show()
	# 