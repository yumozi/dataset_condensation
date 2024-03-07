import os
import copy
import argparse
import torch
import torch.nn as nn
import pdb
import sys
import numpy as np
from utils import get_dataset, epoch, epoch_alt, get_network, TensorDataset, Attack, ParamAttack, NormalizeByChannelMeanStd, get_daparam
import torchattacks
import time

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--real_dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--real_data_path', type=str, default='data', help='real dataset path')
    parser.add_argument('--syn_dataset', type=str, default='CIFAR10', help='original dataset of synthetic data')
    parser.add_argument('--train_data_type', type=str, default='syn', help='train model with syn or real data')
    parser.add_argument('--syn_data_type', type=str, default='Attack', help='which synthetic data to use, Attack/DC/DSA')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--train_epoch', type=int, default=50, help='epochs to train net for real data (syn data is constant)')
    parser.add_argument('--trials', type=int, default=1, help='trials to test net')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_test', type=int, default=256, help='batch size for testing networks')
    parser.add_argument('--attack_eval', type=bool, default=False, help='whether to perform attack during eval')
    parser.add_argument('--attack_strategy', type=str, default='pgd', help='type of adversarial attack')
    parser.add_argument('--transform', type=bool, default=False, help='whether to apply transformation for training')
    parser.add_argument('--aux_bn', type=bool, default=False, help='whether to apply auxiliary BN for training')
    parser.add_argument('--alp_embed', type=bool, default=False, help='whether to apply adversarial logit pairing for training')
    parser.add_argument('--alp_live', type=bool, default=False, help='whether to apply adversarial logit pairing for training')
    parser.add_argument('--ipc', type=int, default=10, help='images per class')
    parser.add_argument('--attack_train', type=bool, default=False, help='whether to perform attack during train')
    # parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    # parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    # parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    # parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    # parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    # parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa = False

    # If aux_bn is True, also load Attacked Distilled Data
    if args.aux_bn or args.alp_embed or args.alp_live:
        args.aux_data_path = f'./result/res_Attack_{args.syn_dataset}_ConvNet_{args.ipc}ipc.pt'
        if not os.path.exists(args.aux_data_path):
            print(f'Error: Attack data not found.')
            sys.exit(1)

        atk_syn_data = torch.load(args.aux_data_path)
        atk_syn_images = atk_syn_data['data'][0][0]
        atk_syn_labels = atk_syn_data['data'][0][1]

        # args.model = 'ConvNetBN'
        if args.syn_data_type == 'Attack':
            print('Error: Attacked data is already loaded as Aux data. Please use other types of synthetic data.')
            sys.exit(1)

    if not os.path.exists(args.real_data_path):
        print(f'Error: {args.real_data_path} not found.')
        sys.exit(1)

    args.syn_data_path = f'./result/res_{args.syn_data_type}_{args.syn_dataset}_ConvNet_{args.ipc}ipc.pt'
    if not os.path.exists(args.syn_data_path):
        print(f'Error: {args.syn_data_path} not found.')
        sys.exit(1)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    args.attack_param = ParamAttack()
    if args.train_data_type == 'real':
        args.syn_data_type = 'no attack'

    if args.train_data_type == 'syn':
        args.train_epoch = 1000

    # Load Distilled Data
    syn_data = torch.load(args.syn_data_path)
    syn_images = syn_data['data'][0][0]
    syn_labels = syn_data['data'][0][1]
    channel = 3
    num_classes = 10
    im_size = (32, 32)
    if args.syn_dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
    elif args.syn_dataset == 'CIFAR100':
        num_classes = 100
    elif args.syn_dataset == 'ImageNette':
        num_classes = 10
        im_size = (160, 160)

    # Load Real Data (only need info if using real data to train)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.real_dataset, args.real_data_path)
    if args.train_data_type == 'real':
        real_train_images = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        real_train_labels = [dst_train[i][1] for i in range(len(dst_train))]
        real_train_images = torch.cat(real_train_images, dim=0).to(args.device)
        real_train_labels = torch.tensor(real_train_labels, dtype=torch.long, device=args.device)

    real_test_images = [torch.unsqueeze(dst_test[i][0], dim=0) for i in range(len(dst_test))]
    real_test_labels = [dst_test[i][1] for i in range(len(dst_test))]
    real_test_images = torch.cat(real_test_images, dim=0).to(args.device)
    real_test_labels = torch.tensor(real_test_labels, dtype=torch.long, device=args.device)

    normalize = NormalizeByChannelMeanStd(mean=mean, std=std)

    acc_list = []
    time_list = []
    for trial in range(args.trials):

        # Setup net
        net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model

        # torchattack fix
        net = nn.Sequential(normalize, net)

        net.train()
        net_parameters = list(net.parameters())
        if args.train_data_type == 'syn':
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net, momentum=0.9, weight_decay=0.0005)  # optimizer_img for synthetic data
        else: 
            optimizer_net = torch.optim.Adam(net.parameters(), lr=args.lr_net)  # use Adam for real data
        optimizer_net.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        loss_avg = 0

        # Setup training data
        if args.train_data_type == 'syn':
            images_train, labels_train = copy.deepcopy(syn_images.detach()), copy.deepcopy(syn_labels.detach())
        else:
            images_train, labels_train = copy.deepcopy(real_train_images.detach()), copy.deepcopy(real_train_labels.detach()) 

        data_train = TensorDataset(images_train, labels_train)
        trainloader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_train, shuffle=False, num_workers=0) # shuffle must be false for logit

        if args.aux_bn or args.alp_embed or args.alp_live:
            # assume args.train_data_type == 'syn'
            atk_images_train, atk_labels_train = copy.deepcopy(atk_syn_images.detach()), copy.deepcopy(atk_syn_labels.detach())
            atk_data_train = TensorDataset(atk_images_train, atk_labels_train)
            atk_trainloader = torch.utils.data.DataLoader(atk_data_train, batch_size=args.batch_train, shuffle=False, num_workers=0) # shuffle must be false for logit

        model_eval = [args.model]
        args.dc_aug_param = get_daparam(args.real_dataset, args.model, model_eval, args.ipc)

        # Train net
        time_start = time.time()
        for e in range(args.train_epoch):
            if not args.aux_bn and not args.alp_embed and not args.alp_live:
                loss, acc = epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True)
            else:
                loss, acc = epoch_alt('train', trainloader, atk_trainloader, net, optimizer_net, criterion, args, aug = False)

        # Setup test data (attack handled in epoch function)
        images_test, labels_test = copy.deepcopy(real_test_images.detach()), copy.deepcopy(real_test_labels.detach()) 
        data_test = TensorDataset(images_test, labels_test)
        testloader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_test, shuffle=True, num_workers=0)
        time_end = time.time()
        
        # Test net
        loss, acc = epoch('test', testloader, net, optimizer_net, criterion, args, aug = False)
        print(f'Trial {trial}: acc = {acc}')
        acc_list.append(acc)
        time_list.append(time_end - time_start)

    if args.attack_eval:
        print(f'Trained with {args.train_data_type} data ({args.syn_data_type}), tested on {args.attack_strategy} attacked {args.real_dataset} dataset.')
    else:
        print(f'Trained with {args.train_data_type} data ({args.syn_data_type}), tested on normal {args.real_dataset} dataset.')

    print(f'Average accuracy over {args.trials} trials is {round(np.mean(acc_list) * 100, 1)}±{round(np.std(acc_list) * 100, 1)}')
    print(f'Average time over {args.trials} trials is {round(np.mean(time_list), 1)}±{round(np.std(time_list), 1)}')

if __name__ == '__main__':
    main()
