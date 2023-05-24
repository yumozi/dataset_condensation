"""
4 experiment settings:
[train a model on normal / robust distilled data] x [test curvature on real data / the same distilled data the model was trained on]
controlled by args.syn_method and args.test_syn respectively
Hypothetical result:
(1) real data has flatter curvature to a model trained on robust distilled data than a model trained on standard distilled data (validated with a sample size of 100)
(2) real data and standard distilled data have similar curvature profile to a model trained on standard distilled data
(3) real data has flatter curvature than robust distilled data to a model trained on robust distilled data
"""

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
    parser.add_argument('--syn_method', type=str, default='DC', help='which synthetic data to use, Attack/DC/DSA/CURE')
    parser.add_argument('--test_syn', action='store_true', help='whether to test curvature on the synthetic set')
    parser.add_argument('--ipc', type=int, default=10, help='images per class')

    # Training
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--train_epoch', type=int, default=1000,
                        help='epochs to train net (recommend 50 for real data and 1000 for synthetic data)')
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
    H = H.reshape((3 * 32 * 32, 3 * 32 * 32))
    H = H.cpu()
    eigenvalues = np.linalg.eigvalsh(H)  # Compute eigenvalues
    return eigenvalues


# Get args
args = parse_args()

# Load the dataset
print('==> Preparing data..')
channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset,
                                                                                                     "data")
real_train_images = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
real_train_labels = [dst_train[i][1] for i in range(len(dst_train))]
real_train_images = torch.cat(real_train_images, dim=0).to(args.device)
real_train_labels = torch.tensor(real_train_labels, dtype=torch.long, device=args.device)

real_test_images = [torch.unsqueeze(dst_test[i][0], dim=0) for i in range(len(dst_test))]
real_test_labels = [dst_test[i][1] for i in range(len(dst_test))]
real_test_images = torch.cat(real_test_images, dim=0).to(args.device)
real_test_labels = torch.tensor(real_test_labels, dtype=torch.long, device=args.device)

args.train_epoch = 1000  # Always train on the syn data
syn_data_path = f'./result/res_{args.syn_method}_{args.dataset}_ConvNet_{args.ipc}ipc.pt'
syn_data = torch.load(syn_data_path)
syn_images = syn_data['data'][0][0]
syn_labels = syn_data['data'][0][1]
images_train, labels_train = copy.deepcopy(syn_images.detach()), copy.deepcopy(syn_labels.detach())
if args.dataset == 'CIFAR10':
    channel = 3
    num_classes = 10
    im_size = (32, 32)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model

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
trainloader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_train, shuffle=True,
                                          num_workers=0)  # shuffle must be false for logit

# Train net
print('==> Training network..')
for e in range(args.train_epoch):
    if e % 10 == 0:
        print(f'Epoch {e}')
    loss, acc = epoch('train', trainloader, net, optimizer_net, criterion, args, aug=False)

if args.test_syn:
    data_test = data_train
else:
    data_test = TensorDataset(real_test_images, real_test_labels)

# Create a random sampler
random_sampler = torch.utils.data.RandomSampler(data_test, num_samples=args.samples, replacement=False)

# Create a data loader
testloader = torch.utils.data.DataLoader(
    data_test,
    batch_size=1,
    sampler=random_sampler,
    num_workers=0,
    pin_memory=False
)

curvature_averages = []

# Compute curvature profile for each sampled image
print('==> Computing curvature profile..')
for image, label in testloader:
    image = image.to(args.device)
    label = label.to(args.device)

    eigenvalues = compute_curvature_profile(net, image, label)
    curvature_averages.append(eigenvalues)


# Compute the average curvature profile
print('==> Computing average..')
sorted_averages = np.sort(np.array(curvature_averages)).mean(axis=0).tolist()

# Sort the averages from low to high
# sorted_averages = sorted(curvature_profile)

# Plot the averages in a line graph
plt.plot(sorted_averages)
plt.xlabel("Index")
plt.ylabel("Average Eigenvalue")
plt.title("Curvature Averages")

# Bound the y-axis
axes = plt.gca()
axes.set_ylim([-0.6, 1])

#plt.show()
plt.savefig(f"{args.syn_method}.png")

#if args.dataset_type == 'real':
    #filename = f"real_{args.dataset}_{args.ipc}ipc.csv"
#else:
filename = f"{args.syn_method}_{str(int(args.test_syn))}_{args.dataset}_{args.ipc}ipc.csv"

# Open the file in write mode
with open(filename, "w", newline="") as file:
    writer = csv.writer(file)

    # Write the numbers to the CSV file
    writer.writerow(sorted_averages)

print(f"Numbers successfully written to {filename}")