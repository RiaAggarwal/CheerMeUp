import torchvision.models as models
import torch.nn as nn

class Classifier(nn.Module):
	def __init__(self, num_classes):
		super(Classifier, self).__init__()
		self.model = models.alexnet(pretrained=True)

		self.fc = nn.Linear(256, num_classes)
		nn.init.xavier_uniform_(self.fc.weight, .1)
		nn.init.constant_(self.fc.bias, 0.)

	def forward(self, x):
		x = self.fc(self.model.features(x))
		return x