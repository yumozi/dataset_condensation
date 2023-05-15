import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt 
import numpy as np
import torch.nn as nn
import argparse
import os
import copy
from utils import get_dataset, get_network, epoch, NormalizeByChannelMeanStd, TensorDataset
import pdb
import csv


def parse_args():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # Model and Dataset
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--dataset_type', type=str, default='syn', help='real/syn, determines where to look for data')
    parser.add_argument('--syn_method', type=str, default='DC', help='which synthetic data to use, Attack/DC/DSA/CURE')
    parser.add_argument('--ipc', type=int, default=10, help='images per class')

    # Training
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--train_epoch', type=int, default=50, help='epochs to train net (recommend 50 for real data and 1000 for synthetic data)')
    parser.add_argument('--attack_eval', type=bool, default=False, help='whether to perform attack during eval')

    # Eigen Experiment
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--samples', type=int, default=1, help='how many random samples to take')
    parser.add_argument('--h', type=float, default=1, help='finite difference')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args

# Function to compute the curvature profile
def compute_curvature_profile(model, image, label):
    # Set the model to evaluation mode
    model.eval()

    # Change size from (1, 3, 32, 32) to (3, 32, 32)
    image = image.squeeze(0)

    def loss_function(image):
        logits = model(image)
        return F.cross_entropy(logits, label)

    H = autograd.functional.hessian(loss_function, image)
    H = H.reshape((3*32*32, 3*32*32))
    H = H.cpu()
    eigenvalues, _ = np.linalg.eig(H)  # Compute eigenvalues
    return eigenvalues

# Get args
args = parse_args()

# Load the dataset
print('==> Preparing data..')
if args.dataset_type == 'real':
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, "data")

    images_train = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_train = [dst_train[i][1] for i in range(len(dst_train))]
    images_train = torch.cat(images_train, dim=0).to(args.device)
    labels_train = torch.tensor(labels_train, dtype=torch.long, device=args.device)
    
    images = [torch.unsqueeze(dst_test[i][0], dim=0) for i in range(len(dst_test))]
    labels = [dst_test[i][1] for i in range(len(dst_test))]
    images = torch.cat(images, dim=0).to(args.device)
    labels = torch.tensor(labels, dtype=torch.long, device=args.device)
else:
    syn_data_path = f'./result/res_{args.syn_method}_{args.dataset}_ConvNet_{args.ipc}ipc.pt'
    syn_data = torch.load(syn_data_path)
    images = syn_data['data'][0][0]
    labels = syn_data['data'][0][1]
    images_train, labels_train = copy.deepcopy(images.detach()), copy.deepcopy(labels.detach())
    if args.dataset == 'CIFAR10':
        channel = 3
        num_classes = 10
        im_size = (32, 32)
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
dataset = TensorDataset(images, labels)

net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model

# torchattack fix
normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
net = nn.Sequential(normalize, net)
net.to(args.device)

net.train()
net_parameters = list(net.parameters())
optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
criterion = nn.CrossEntropyLoss().to(args.device)
loss_avg = 0

data_train = TensorDataset(images_train, labels_train)
trainloader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_train, shuffle=True, num_workers=0) # shuffle must be false for logit

# Train net
print('==> Training network..')
for e in range(args.train_epoch):
    if e % 10 == 0:
        print(f'Epoch {e}')
    loss, acc = epoch('train', trainloader, net, optimizer_net, criterion, args, aug = False)

# Create a random sampler
random_sampler = torch.utils.data.RandomSampler(dataset, num_samples=args.samples, replacement=False)

# Create a data loader
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    sampler=random_sampler,
    num_workers=0,
    pin_memory=False
)

curvature_averages = []

# Compute curvature profile for each sampled image
print('==> Computing curvature profile..')
for image, label in data_loader:
    image = image.to(args.device)
    label = label.to(args.device)

    eigenvalues = compute_curvature_profile(net, image, label)
    curvature_averages.append(eigenvalues)

# Compute the average curvature profile
print('==> Computing average..')
curvature_profile = []
for i in range(len(curvature_averages[0])):
    total = 0
    for j in range(args.samples):
        total += curvature_averages[j][i]
    curvature_profile.append(total / args.samples)

# Sort the averages from low to high
sorted_averages = sorted(curvature_profile)

# Plot the averages in a line graph
plt.plot(sorted_averages)
plt.xlabel("Index")
plt.ylabel("Average Eigenvalue")
plt.title("Curvature Averages")

# Bound the y-axis
axes = plt.gca()
axes.set_ylim([-0.001, 0.001])

plt.show()

if args.dataset_type == 'real':
    filename = f"real_{args.dataset}_{args.ipc}ipc.csv"
else:
    filename = f"{args.syn_method}_{args.dataset}_{args.ipc}ipc.csv"

# Open the file in write mode
with open(filename, "w", newline="") as file:
    writer = csv.writer(file)

    # Write the numbers to the CSV file
    writer.writerow(sorted_averages)

print(f"Numbers successfully written to {filename}")