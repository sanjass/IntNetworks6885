import os
import json
from viz import analyze_video
from utils import build_recursive_case_paths
from data_preprocessing import binary_indices, mse_indices, categorical_indices, index_mapping
import torch.nn.functional as F
from hyperparams import IDX_2_SHAPE


def compare_objects_valid(obj_true, obj_pred):

	# for key, val in binary_indices.items():
	# 	if key == "is_present":
	# 		presence_log = F.log_softmax(obj_pred[val[0]:val[-1]+1], dim=0) 
	# 		presence_bit = 1 if presence_log[0] > presence_log[1] else 0
	# 		ground_truth_presence = 1 if obj_true[val[0]:val[-1]+1][0] == 1 else 0
	# 		if presence_bit != ground_truth_presence:
	# 			reason = f"Predicted presence bit is {presence_bit}, but ground truth is {ground_truth_presence}"
	# 			return False, reason

	# 	elif key == "is_visible":
	# 		visible_log = F.log_softmax(obj_pred[val[0]:val[-1]+1], dim=0) 
	# 		visible_bit = 1 if visible_log[0] > visible_log[1] else 0
	# 		ground_truth_visible = 1 if obj_true[val[0]:val[-1]+1][0] == 1 else 0
	# 		if visible_bit != ground_truth_visible:
	# 			reason = f"Predicted visible bit is {visible_bit}, but ground truth is {ground_truth_visible}"
	# 			return False, reason
	# 	elif key == "is_occluder":
	# 		continue
	# 	else:
	# 		raise ValueError("Should never get here")

	# for key, val in categorical_indices.items():
	# 	if key == "shape":
	# 		shape_log = F.log_softmax(obj_pred[val[0]:val[-1]+1], dim=0) 
	# 		_, shape_bit = shape_log.max(dim=0)
	# 		_, ground_truth_shape = obj_true[val[0]:val[-1]+1].max(dim=0)
	# 		if shape_bit != ground_truth_shape:
	# 			reason = f"Predicted shape is {IDX_2_SHAPE[shape_bit.item()]}, but ground truth is {IDX_2_SHAPE[ground_truth_shape.item()]}"
	# 			return False, reason
	# 	else:
	# 		raise ValueError("Should never get here")

	for key, val in mse_indices.items():
		if "location" in key:
			true_loc = obj_true[val[0]]
			pred_loc = obj_pred[val[0]]
			abs_diff = abs(true_loc - pred_loc)
			if abs_diff > 100:
				reason = f"Predicted {key} is {pred_loc}, but ground truth is {true_loc}"
				return False, reason

	return True, "No anomalies detected"






def dev_meta():
	case_names = build_recursive_case_paths("../intphys/dev_meta", [])
	meta_info = {}
	for video_path in case_names:
		ground_truths, predictions = analyze_video(video_path, use_ground_truth=True)
		ground_truth_label =json.load(open(video_path +"/status.json", 'r'))["header"]["is_possible"]
		possible = True # by default a video is possible
		print("video: ", video_path)

		# for key, val in mse_indices.items():
		# 	print("\nkey: ", key)
		# 	print("grouns truths: ", ground_truths[1][0][0][val[0]].item())
		# 	print("predictions: ", predictions[1][0][0][val[0]].item())
		# break

		for frame_idx in range(len(ground_truths)):
			for obj_idx in range(5): # max objects
				obj_true = ground_truths[frame_idx][0][obj_idx]
				obj_pred = predictions[frame_idx][0][obj_idx]
				# print("\nframe idx: ", frame_idx)
				# print("Obj index: ", obj_idx)
				pred_label, reason = compare_objects_valid(obj_true, obj_pred)
				if pred_label is False:
					possible = False
					meta_info = {"frame":frame_idx, "obj":obj_idx}
					break
			if possible == False:
				break
				
				
		

		if possible == ground_truth_label:
			print(f"Correct prediction! Predicted and ground truth are {possible}")
			print("Reason: ", reason)
			print("Meta info: ", meta_info)
		else:
			print(f"Incorrect, predicted {possible}, but ground truth label is {ground_truth_label}")
			print("Reason: ", reason)
			print("Meta info: ", meta_info)

		break



if __name__ == "__main__":
	dev_meta()