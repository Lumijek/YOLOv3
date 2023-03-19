import torch
import torch.nn as nn
import torch.nn.functional as F
from config import anchors

def get_params(model):
	print(sum(p.numel() for p in model.parameters()))
class ConvLayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, alpha=0.1, bn=True):
		super().__init__()
		self.bn = bn
		if padding == None:
			padding = kernel_size // 2
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
		self.bn = nn.BatchNorm2d(out_channels)
		self.activation = nn.LeakyReLU(alpha)

	def forward(self, x):
		if self.bn:
			return self.activation(self.bn(self.conv(x)))
		else:
			return self.conv(x)

# custom residual block for darknet-53
class ResidualBlock(nn.Module):
	def __init__(self, in_channels, mid_channels, kernel_size1=1, kernel_size2=3, num_layers=1):
		super().__init__()

		self.layers = nn.ModuleList()
		for _ in range(num_layers):
			layer = nn.Sequential(
				ConvLayer(in_channels, mid_channels, kernel_size1),
				ConvLayer(mid_channels, in_channels, kernel_size2)
			)
			self.layers.append(layer)

	def forward(self, x):
		for layer in self.layers:
			x = x + layer(x)
		return x

class DarkNet53(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Sequential(
			ConvLayer(3, 32, 3),
			ConvLayer(32, 64, 3, stride=2)
		)
		self.residual1 = ResidualBlock(64, 32, num_layers=1)

		self.conv2 = ConvLayer(64, 128, 3, stride=2)
		self.residual2 = ResidualBlock(128, 64, num_layers=2)

		self.conv3 = ConvLayer(128, 256, 3, stride=2)
		self.residual3 = ResidualBlock(256, 128, num_layers=8)

		self.conv4 = ConvLayer(256, 512, 3, stride=2)
		self.residual4 = ResidualBlock(512, 256, num_layers=8)

		self.conv5 = ConvLayer(512, 1024, 3, stride=2)
		self.residual5 = ResidualBlock(1024, 512, num_layers=4)

	def forward(self, x):
		x = self.conv1(x)
		x = self.residual1(x)

		x = self.conv2(x)
		x = self.residual2(x)

		x = self.conv3(x)
		out1 = self.residual3(x)

		x = self.conv4(out1)
		out2 = self.residual4(x)

		x = self.conv5(out2)
		out3 = self.residual5(x)

		return out1, out2, out3

class YOLOv3(nn.Module):
	def __init__(self, num_classes=20):
		super().__init__()
		self.num_classes = num_classes

		self.backbone = DarkNet53()

		self.detector1 = nn.Sequential(
			ConvLayer(1024, 512, 1),
			ConvLayer(512, 1024, 3),
			ConvLayer(1024, 512, 1),
			ConvLayer(512, 1024, 3),
			ConvLayer(1024, 512, 1)
		)
		self.out1 = nn.Sequential(
			ConvLayer(512, 1024, 3),
			ConvLayer(1024, 3 * (5 + num_classes), 1, bn=False)

		)

		self.upsample1 = nn.Sequential(
			nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1)
		)

		self.detector2 = nn.Sequential(
			ConvLayer(1024, 512, 1),
			ConvLayer(512, 1024, 3),
			ConvLayer(1024, 512, 1),
			ConvLayer(512, 1024, 3),
			ConvLayer(1024, 512, 1)
		)
		self.out2 = nn.Sequential(
			ConvLayer(512, 1024, 3),
			ConvLayer(1024, 3 * (5 + num_classes), 1, bn=False)

		)

		self.upsample2 = nn.Sequential(
			nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1)
		)

		self.detector3 = nn.Sequential(
			ConvLayer(768, 512, 1),
			ConvLayer(512, 1024, 3),
			ConvLayer(1024, 512, 1),
			ConvLayer(512, 1024, 3),
			ConvLayer(1024, 512, 1)
		)
		self.out3 = nn.Sequential(
			ConvLayer(512, 1024, 3),
			ConvLayer(1024, 3 * (5 + num_classes), 1, bn=False)

		)

	def forward(self, x):
		out1, out2, out3 = self.backbone(x)

		cat1 = self.detector1(out3)
		x1 = self.out1(cat1)

		cat1 = torch.cat([self.upsample1(cat1), out2], dim=1)

		cat2 = self.detector2(cat1)
		x2 = self.out2(cat2)

		cat2 = torch.cat([self.upsample2(cat2), out1], dim=1)

		cat3 = self.detector3(cat2)
		x3 = self.out3(cat3)


		x1 = x1.reshape(-1, 3, (5 + self.num_classes), x1.shape[-1], x1.shape[-1])
		x1 = x1.permute(0, 3, 4, 1, 2).contiguous()

		x2 = x2.reshape(-1, 3, (5 + self.num_classes), x2.shape[-1], x2.shape[-1])
		x2 = x2.permute(0, 3, 4, 1, 2).contiguous()

		x3 = x3.reshape(-1, 3, (5 + self.num_classes), x3.shape[-1], x3.shape[-1])
		x3 = x3.permute(0, 3, 4, 1, 2).contiguous()

		return x1, x2, x3



