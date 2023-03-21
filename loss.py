import torch
import torch.nn as nn
from config import anchors
import torchvision.ops as bops

def iou(bbox1, bbox2):
	bbox1 = bbox1.clone()
	bbox2 = bbox2.clone()
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

	out = torch.diagonal(bops.box_iou(bbox1, bbox2)).unsqueeze(0)
	return out

class Yolov3Loss(nn.Module):
	def __init__(self, lambda_coord=5, lambda_noobj=0.5, lambda_class=1):
		super().__init__()
		self.lambda_coord = lambda_coord
		self.lambda_noobj = lambda_noobj
		self.lambda_class = lambda_class

		self.register_buffer("anchors", anchors.reshape(3, 3, 2))
		self.mse = nn.MSELoss(reduction="mean")
		self.cel = nn.CrossEntropyLoss(reduction="sum")
		self.bce = nn.BCEWithLogitsLoss(reduction='mean')

	def forward(self, predictions, labels, scale_idx):
		print("")
		batch_size = predictions.shape[0]
		scale_anchors = self.anchors[scale_idx]

		S = predictions.shape[1]
		object_inds = labels[..., 4] == 1
		*_, anchors_indices = torch.where(object_inds)
		predictions[:, :, :, :, 0:2] = torch.sigmoid(predictions[:, :, :, :, 0:2])
		predictions[:, :, :, :, 2:4] = scale_anchors * torch.exp(predictions[:, :, :, :, 2:4])

		obj_preds = predictions[object_inds]
		obj_labels = labels[object_inds]

		noobj_preds = predictions[~object_inds]
		noobj_labels = labels[~object_inds]

		if obj_preds.shape[0] == 0:
			loss_noobj_conf = self.bce(noobj_preds[:, 4], noobj_labels[:, 4])
			loss_confidence = (loss_noobj_conf * self.lambda_noobj) / batch_size
			return loss_confidence, torch.tensor(0), loss_confidence, torch.tensor(0)

		# LOSS CALCULATION

		loss_xy = self.mse(obj_preds[:, :2], obj_labels[:, :2])
		loss_wh = self.mse(torch.sqrt(obj_preds[:, 2:4]), torch.sqrt(obj_labels[:, 2:4]))

		loss_obj_conf = self.bce(obj_preds[:, 4], obj_labels[:, 4])

		loss_noobj_conf = self.bce(noobj_preds[:, 4], noobj_labels[:, 4])

		loss_class = self.cel(obj_preds[:, 5:], obj_labels[:, 5].long())

		loss_coord = (self.lambda_coord * (loss_xy + loss_wh))# / batch_size
		loss_confidence = (loss_obj_conf + loss_noobj_conf * self.lambda_noobj)# / batch_size
		loss_class = (self.lambda_class * loss_class) / batch_size

		loss = loss_coord + loss_confidence + loss_class

		return loss, loss_coord, loss_confidence, loss_class


