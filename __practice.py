from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


from baseline import BaseLineClassifier




print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def set_paramseter_requires_grad(model, feature_extracting):
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad=False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
	model_ft = None
	input_size = 0
	if model_name =='resnet':
		model_ft = models.resnet18(pretrained=use_pretrained)
		set_paramseter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.fc.in_features
		model_ft.fc = nn.Linear(num_ftrs, num_classes)
		input_size = 224
	elif model_name == 'vgg':
		model_ft = models.vgg11_bn(pretrained=use_pretrained)
		set_paramseter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6]=nn.Linear(num_ftrs,num_classes)
		input_size =224
	return model_ft, input_size

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
	since = time.time()
	val_acc_history = []
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	for epoch in range(num_epochs):
		print("Epoch {}/{}".format(epoch, num_epochs-1))
		print('-'*10)

		for phase in ['train','val']:
			if phase == 'train':
				model.train()
			else:
				model.eval()
			running_loss = 0.0
			running_corrects = 0

			for inputs, labels in dataloaders[phase]:
				print(inputs.shape)
				inputs = inputs.to(device)
				labels = labels.to(device)

				optimizer.zero_grad()

				with torch.set_grad_enabled(phase=='train'):
					if phase=='train':
						outputs = model(inputs)
						loss = criterion(outputs, labels)
					print(inputs.shape)
					print(outputs.shape)
					print(labels.shape)
					_, preds = torch.max(outputs, 1)
					print(preds.shape)
					
					if phase=='train':
						loss.backward()
						optimizer.step()
				running_loss += loss.item()*inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
			if phase == 'val':
				val_acc_history.append(epoch_acc)
		print()
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))
	# load best model weights
	model.load_state_dict(best_model_wts)
	return model, val_acc_history

# if __name__ == "__main__":
	# model = BaseLineClassifier()
	# batch_frame_seq = torch.randn(5, 7 ,128)
	# model(batch_frame_seq)

data_dir = "../data/hymenoptera_data"
model_name = "vgg"
num_classes = 2
batch_size = 8
num_epochs = 15
feature_extract = True

model_ft, input_size = initialize_model(model_name ,num_classes, feature_extract, use_pretrained=True)
print(model_ft)

data_transforms = {
'train': transforms.Compose([
		transforms.RandomResizedCrop(input_size),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
'val': transforms.Compose([
	transforms.Resize(input_size),
	transforms.CenterCrop(input_size),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
}

print("initializing datasets and dataloaders")

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x]) for x in ['train','val']}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
print("params to update")
if feature_extract:
	params_to_update = []
	for name, param in model_ft.named_parameters():
		if param.requires_grad == True:
			params_to_update.append(param)
			print("\t",name)
else:
	pass

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


criterion = nn.CrossEntropyLoss()

model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)