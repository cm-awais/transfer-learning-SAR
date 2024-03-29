# Imports

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import multiprocessing as mp
from torchvision import models
from copy import deepcopy
import warnings
import csv


warnings.filterwarnings('ignore')


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
results = ""

# Define data transformations
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def cal_scores(targets, predictions, check=False):
  if check:
    true_labels = [int(t.item()) for t in targets]  # Extract integer values
    predicted_labels = [int(p.item()) for p in predictions]

  class_info = {0:[],
                1:[],
                2:[]}
  scores = []

  for id, val in enumerate(true_labels):
    if val == 0:
      class_info[0].append((val, predicted_labels[id]))
    if val == 1:
      class_info[1].append((val, predicted_labels[id]))
    if val == 2:
      class_info[2].append((val, predicted_labels[id]))

  class_metrics = {}

  overall_accuracy = accuracy_score(true_labels, predicted_labels)

  # Calculate overall precision, recall, and F1-score (weighted average)
  overall_precision = precision_score(true_labels, predicted_labels, average='weighted')
  overall_recall = recall_score(true_labels, predicted_labels, average='weighted')
  overall_f1 = f1_score(true_labels, predicted_labels, average='weighted')
  scores.extend([overall_accuracy, overall_precision, overall_recall, overall_f1])

  for j in range(3):
    tr_lab = []
    pr_lab = []
    for items in class_info[j]:
      tr_lab.append(items[0])
      pr_lab.append(items[1])
    class_accuracy = accuracy_score(tr_lab, pr_lab)
    class_precision = precision_score(tr_lab, pr_lab, average='weighted')
    class_recall = recall_score(tr_lab, pr_lab, average='weighted')
    class_f1 = f1_score(tr_lab, pr_lab, average='weighted')
    scores.extend([class_accuracy, class_precision, class_recall, class_f1])

    class_metrics[j] = {
        'accuracy': class_accuracy,
        'precision': class_precision,
        'recall': class_recall,
        'f1': class_f1
    }
  # Return dictionary containing all metrics
  return {
      'class_metrics': class_metrics,
      'overall_accuracy': overall_accuracy,
      'overall_precision': overall_precision,
      'overall_recall': overall_recall,
      'overall_f1': overall_f1
  }, scores


# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    model.to(device)
    saving_string = ""
    correct = 0
    total = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            # print(data.shape, target.shape)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted)
            targets.extend(target)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    saving_string += f"Accuracy: {accuracy:.2f}% \n"
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    dicrt, scores = cal_scores(predictions=predictions, targets=targets, check=True)
    print(dicrt)
    return saving_string, scores

def load_data(address, batch_size=64, train=True):
  # Load Fusar dataset
  if train:
    dataset = ImageFolder(root=address, transform=transform_train)
  else: 
    dataset = ImageFolder(root=address, transform=transform_test)

  # Create a dictionary of class names
  class_names = {i: classname for i, classname in enumerate(dataset.classes)}

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2,  # Experiment with different values as recommended above
                            # pin_memory=False, # if torch.cuda.is_available() else False,
                            persistent_workers=True)
  print("Top classes indices:", class_names)

  return data_loader

class VGGModel(nn.Module):
  def __init__(self, pretrained=False):
    super(VGGModel, self).__init__()
    self.features = models.vgg16(pretrained=pretrained).features  # Use VGG16 features
    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Global Average Pooling
    self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 3)  # 3 output classes
        )

  def forward(self, x):
    x = self.features(x)
    # print(x.shape)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # print(x.shape)
    x = self.classifier(x)
    return x

class FineTunedVGG(nn.Module):
  def __init__(self, pretrained=True):
    super(FineTunedVGG, self).__init__()
    self.features = models.vgg16(pretrained=pretrained)
    for param in self.features.parameters():
      param.requires_grad = False  # Freeze pre-trained layers

    # self.classifier = nn.Sequential(*list(self.features.classifier.children())[:-1])  # Use all but last layer
    # self.classifier.add_module('final', nn.Linear(self.classifier[-1].in_features, 3))  # Add new final layer
    self.classifier = nn.Sequential(
            nn.Linear(250 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 3)  # 3 output classes
        )

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x
  
class ResNetModel(nn.Module):
  def __init__(self, pretrained=False):
    super(ResNetModel, self).__init__()
    self.features = models.resnet50(pretrained=pretrained)  # Use ResNet50 features
    self.avgpool = nn.AdaptiveAvgPool2d((10, 10))
    self.classifier = nn.Sequential(
      nn.Linear(10 * 10, 4096),  # Adjust based on input size
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 3)
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 1, x.size(1), 1)
    # print(x.shape)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # print(x.shape)
    x = self.classifier(x)
    return x

class FineTunedResNet(nn.Module):
  def __init__(self, pretrained=True):
    super(FineTunedResNet, self).__init__()

    # Load pre-trained ResNet50 model
    self.features = models.resnet50(pretrained=pretrained)

    # Freeze pre-trained layers
    for param in self.features.parameters():
      param.requires_grad = False

    # Replace final layer and adjust for grayscale input
    self.avgpool = nn.AdaptiveAvgPool2d((10, 10))  # Global Average Pooling
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc = nn.Linear(self.features.fc.in_features, 3)  # Replace final layer
    self.classifier = nn.Sequential(
            nn.Linear(10 * 10, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 3)  # 3 output classes
        )

  def forward(self, x):
    # Convert grayscale image to 3-channel tensor (assuming single channel)
    # print(x.size(1))
    if x.size(1) == 1:  # Check if input has 1 channel
      x = x.repeat(1, 3, 1, 1)  # Duplicate grayscale channel 3 times
    # print(x.size(1))

    x = self.features(x)
    x = x.view(x.size(0), 1, x.size(1), 1)
    # print(x.shape)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # print(x.shape)
    x = self.classifier(x)
    return x
  
class ImprovedCNNModel(nn.Module):
  def __init__(self):
    super(ImprovedCNNModel, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),  # Add Batch Normalization for better stability
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(128, 256, kernel_size=3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Linear(4096 * 7 * 7, 4096),  # Adjust for final feature map size
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),  # Adjust dropout probability
      nn.Linear(4096, 3)
    )

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

# Define CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 28 * 28, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 3)  # 3 output classes
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
lf = nn.CrossEntropyLoss()
epochs = 10
l_rate = 0.001
batch_size = 64

fusar_path = "fusar_split"
open_sar_path = "opensarship_splited"
mix_path = "mix_5"

csv_res = []

print("Loading Data")

fusar_train_loader = load_data(fusar_path + "/train", batch_size=batch_size)
fusar_test_loader = load_data(fusar_path + "/test", batch_size=batch_size)

open_sar_train_loader = load_data(open_sar_path + "/train", batch_size=batch_size)
open_sar_test_loader = load_data(open_sar_path + "/test", batch_size=batch_size)

mix_train_loader = load_data(mix_path + "/train", batch_size=batch_size)
mix_test_loader = load_data(mix_path + "/test", batch_size=batch_size)

print("Data Loaded")

# datasets = {"Fusar_ship": [fusar_train_loader,fusar_test_loader],
#             "OpenSARShip": [open_sar_train_loader, open_sar_test_loader],
#             "Mixed": [mix_train_loader, mix_test_loader]}

print("Made a Dictionary")

models = {"CNN": ImprovedCNNModel(),
          "VGG": VGGModel(),
          "Fine_VGG": FineTunedVGG(),
          "ResNet": ResNetModel(),
          "Fine_Resnet": FineTunedResNet()}

# models = {"ResNet": ResNetModel()}

print("Loaded Models")

# models = []

# for dataset_name, dataset_loader in datasets.items():
#     print("Training on ", dataset_name)
#     results += "Training on " + dataset_name + "\n"
#     # mix_path = "mix_5"
#     train_loader_m = dataset_loader[0]
#     test_loader_m = dataset_loader[1]
#     for model_name, model in models.items():
#         print("Training using :" , model_name)
#         results += "Training using "+model_name + "\n"
        
#         n_model = model
#         optimizer = optim.Adam(n_model.parameters(), lr=l_rate)
#         n_model.to(device)

#         # Training loop
#         for epoch in range(epochs):
#             n_model.train()
#             print(epoch)
#             for batch_idx, data in enumerate(train_loader_m):
#                 data, target = data[0].to(device), data[1].to(device)

#                 optimizer.zero_grad()

#                 output = n_model(data)

#                 loss = lf(output, target)
#                 loss.backward()
#                 optimizer.step()

#                 if batch_idx % 57 == 0:
#                     results+=f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader_m)}, Loss: {loss.item()}\n'
#                     print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader_m)}, Loss: {loss.item()}')

#         # Evaluation
#         for dataset_name, data_loaders in datasets.items():
#             print("Evaluating: ", dataset_name)
#             results += f"Testing {model_name} on {dataset_name} \n"
#             str_results, csv_scores = evaluate_model(n_model, data_loaders[1])
#             results += str_results
#             csv_res.append(csv_scores)

results += "Pre-Training on datasets \n"

datasets = {"Fusar_ship+openSar+mix": [[fusar_train_loader, open_sar_train_loader, mix_train_loader],fusar_test_loader],
            "OpenSARShip+fusar+mix": [[open_sar_train_loader, fusar_train_loader, mix_train_loader], open_sar_test_loader],
            "Mixed+fusar+open_sar": [[mix_train_loader, fusar_train_loader, open_sar_train_loader], mix_test_loader]}

# test_sets = [fusar_test_loader, open_sar_test_loader, mix_test_loader]

for dataset_name, dataset_loader in datasets.items():
    print("Training on ", dataset_name)
    results += "Training on " + dataset_name + "\n"
    # mix_path = "mix_5"
    train_loader_m = dataset_loader[0]
    # test_loader_m = dataset_loader[1]
    for model_name, model in models.items():
        print("Training using :" , model_name)
        results += "Training using "+model_name + "\n"
        
        n_model = model
        optimizer = optim.Adam(n_model.parameters(), lr=l_rate)
        n_model.to(device)

        # Training loop
        for epoch in range(epochs):
            n_model.train()
            print(epoch)
            for batch_idx, data in enumerate(train_loader_m[0]):
                data, target = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                output = n_model(data)

                loss = lf(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 57 == 0:
                    results+=f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader_m[0])}, Loss: {loss.item()}\n'
                    print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader_m[0])}, Loss: {loss.item()}')

        # Evaluation
        for dataset_name, data_loaders in datasets.items():
            if data_loaders[1]:
              print("Evaluating: ", dataset_name)
              results += f"Testing {model_name} on {dataset_name} \n"
              str_results, csv_scores = evaluate_model(n_model, data_loaders[1])
              results += str_results
              csv_res.append(csv_scores)


        for param in n_model.features.parameters():
            param.requires_grad = False  # Freeze all parameters

        # Unfreeze the final layer (classifier head)
        for param in n_model.classifier.parameters():
            param.requires_grad = True  # Allow gradients for fine-tuning
        
        print(f"Frozen Parameters: {model_name}\n")

        for ds in train_loader_m[1:]:
          model_s = deepcopy(n_model)
          
          results += f"Fine-Tuning step: {model_name}\n"
          print(f"Fine-Tuning step: {model_name}\n")

          # Transfer-Training loop
          for epoch in range(epochs):
              model_s.train()
              print(epoch)
              for batch_idx, data in enumerate(ds):
                  data, target = data[0].to(device), data[1].to(device)

                  optimizer.zero_grad()

                  output = model_s(data)

                  loss = lf(output, target)
                  loss.backward()
                  optimizer.step()

                  if batch_idx % 57 == 0:
                      results+=f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(ds)}, Loss: {loss.item()}\n'
                      print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(ds)}, Loss: {loss.item()}')

          # Evaluation
          for dataset_name, data_loaders in datasets.items():
              if data_loaders[1]:
                print("Evaluating: ", dataset_name)
                results += f"Testing {model_name} on {dataset_name} \n"
                str_results, csv_scores = evaluate_model(model_s, data_loaders[1])
                results += str_results
                csv_res.append(csv_scores)

# Open the file in write mode ("w") and write the string to it
with open("tl_results.txt", "w") as f:
  f.write(results)

fields = ["Accuracy", "Precision", "Recall", "F1", "C-Accuracy", "C-Precision",
          "C-Recall", "C-F1", "F-Accuracy", "F-Precision", "F-Recall", "F-F1",
          "T-Accuracy", "T-Precision", "T-Recall", "T-F1"]

with open('tl_results.csv', 'w') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(csv_res)
