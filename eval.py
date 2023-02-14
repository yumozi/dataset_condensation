import os
import copy
import argparse
import torch
import torch.nn as nn
import pdb
from utils import epoch, get_network, TensorDataset


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--data_type', type=str, default='syn', help='syn or real data')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--train_epoch', type=int, default=50, help='epochs to train net')
    parser.add_argument('--trials', type=int, default=50, help='trials to test net')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    # parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    # parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    # parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    # parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    # parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    # parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(args.data_path):
        print(f'Error: {args.data_path} not found.')
        sys.exit(1)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if args.data_type == 'syn':
        data = torch.load(args.data_path)
        images = data['data'][0][0]
        labels = data['data'][0][1]

        # Hard coded for now
        channel = 3
        num_classes = 10
        im_size = (32, 32)


    acc_total = 0
    for trial in range(args.trials):
        net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
        net.train()
        net_parameters = list(net.parameters())
        optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
        optimizer_net.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        loss_avg = 0

        images_train, labels_train = copy.deepcopy(images.detach()), copy.deepcopy(labels.detach())  # avoid any unaware modification
        data_train = TensorDataset(images_train, labels_train)
        trainloader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
        for e in range(args.train_epoch):
            epoch('train', trainloader, net, optimizer_net, criterion, args, aug = False)

             
        loss, acc = epoch('test', trainloader, net, optimizer_net, criterion, args, aug = False)
        print(f'Trial {trial}: acc = {acc}')
        acc_total += acc

    print(f'Average accuracy over {args.trials} trials is {acc_total / args.trials}')

if __name__ == '__main__':
    main()
