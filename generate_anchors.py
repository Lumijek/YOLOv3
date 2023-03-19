import ast

import torch
import torchvision.ops.boxes as bops

from dataset import *
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches


torch.set_printoptions(10)
@torch.no_grad()
def iou_anchor(bbox1, bbox2):
	bbox1 = bbox1.clone().detach()
	bbox2 = bbox2.clone().detach()
	bbox1 += 1e-14
	bbox2 -= 1e-16
	bb1w = bbox1[:, 2].clone()
	bb1h = bbox1[:, 3].clone()
	bb2w = bbox2[:, 2].clone()
	bb2h = bbox2[:, 3].clone()
	bbox1[:, 2] = bbox1[:, 0] + bb1w / 2
	bbox1[:, 0] = bbox1[:, 0] - bb1w / 2
	bbox1[:, 3] = bbox1[:, 1] + bb1h / 2
	bbox1[:, 1] = bbox1[:, 1] - bb1h / 2

	bbox2[:, 2] = bbox2[:, 0] + bb2w / 2
	bbox2[:, 0] = bbox2[:, 0] - bb2w / 2
	bbox2[:, 3] = bbox2[:, 1] + bb2h / 2
	bbox2[:, 1] = bbox2[:, 1] - bb2h / 2
	#convert to [x1, y1, x2, y2]
	#out = torch.diagonal(bops.box_iou(bbox1, bbox2)).unsqueeze(0)
	out = bops.box_iou(bbox1, bbox2) 
	#out = out.nan_to_num(1)
	return out

def write_to_file(file):
	a = get_dataset(64)
	f = open(file, "w")
	for it, (i, o) in enumerate(a):
		print(it)
		bboxes = o[o[:, :, :, 4] == 1][:, 2:4]
		tl_dims = torch.zeros_like(bboxes)
		bboxes = torch.concat([tl_dims, bboxes], dim=1).tolist()
		for b in bboxes:
			f.write(', '.join(map(str, b)) + "\n")

	f.close()
	print("Done writing bounding boxes to file")

def generate_anchors(iterations, k):
	groups = {i: [] for i in range(k)}
	file = "train_bboxes.txt"
	#write_to_file(file)
	bboxes = []
	with open(file) as f:
		bboxes = torch.tensor([ast.literal_eval(f"[{line[:-1]}]") for line in f.readlines()])
	starting_groups = torch.randint(0, len(bboxes), (k, ))
	start = bboxes[starting_groups]
	bboxes = bboxes
	for i in range(iterations):
		dis = 1 - iou_anchor(bboxes, start)
		closest_cluster = dis.min(dim=1).indices.tolist()
		for ind, n in enumerate(closest_cluster):
			groups[n].append(bboxes[ind])
		for key, value in groups.items():
			v = torch.cat(value, dim=0).reshape(-1, 4)
			new_best = v.mean(dim=0)
			start[key] = new_best
			groups[key] = []
		print("Iteration:", i)
	return start
@torch.no_grad()
def show_image(inp):
	fig, ax = plt.subplots()
	i, o = inp
	o = o[0]
	S = o.shape[0]
	i = i[0].permute(1, 2, 0).numpy()
	outs = nms(o, 0.30, 0.5)
	for o2 in outs:
		#print(o2)
		text = classes[int(o2[-1])] + str(round(float(o2[4]), 4))
		print(o2)
		xmin = float(o2[0])
		ymin = float(o2[1])
		width = float(o2[2])
		height = float(o2[3])
		ax.text(xmin, 1 - ymin, text, c='blue', weight='bold', backgroundcolor='0.75', transform=plt.gca().transAxes)
		rect = patches.Rectangle((xmin * image_size , ymin * image_size), width * image_size, height * image_size, linewidth=1, edgecolor='r', facecolor='none')
		ax.add_patch(rect)

	ax.imshow(i)
	plt.show()
if __name__ == '__main__':
	anchors = generate_anchors(120, 9)[:, 2:].tolist()
	anchors = sorted(anchors, key=lambda x: x[0] * x[1], reverse=True)
	pprint(torch.tensor(anchors))

	fig, ax = plt.subplots()
	for anchor in anchors:
		xmin = 0.5 - anchor[0] / 2
		ymin = 0.5 - anchor[1] / 2
		width = anchor[0]
		height = anchor[1]

		rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
		ax.add_patch(rect)
	plt.show()

