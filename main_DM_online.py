import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import random
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, Attack, ParamAttack

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DM', help='DM/Attack/CURE')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--attack-eval', action='store_true')
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--attack_strategy', type=str, default='pgd', help='attack strategy')
    parser.add_argument('--lamda', type=float, default=100, help='lamda for CURE method')
    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.inner_loop = 10
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.attack_param = ParamAttack()
    args.dsa = True if args.method == 'DM' else False
    args.attack = True if args.method == "Attack" else False
    args.cure = True if args.method == "CURE" else False
    args.dc_aug_param = None

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 1000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []


    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)



        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins'%get_time())
        num_models = 50

        network_pool = [get_network(args.model, channel, num_classes, im_size).to(args.device) for i in range(num_models)]

        for it in range(args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    if args.dsa:
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    elif args.attack:
                        print('Attack strategy: \n', args.attack_strategy)
                        print('Attack parameters: \n', args.attack_param.__dict__)

                    if 'mnist' in args.dataset.lower():
                        atk_args_ls = [{"attack_eval": False},
                                       {"attack_eval": True, "method": "pgd", "eps": 0.3, "alpha": 0.1,
                                        "steps": 10},
                                       {"attack_eval": True, "method": "pgd", "eps": 0.1, "alpha": 0.03,
                                        "steps": 10},
                                       {"attack_eval": True, "method": "pgdl2", "eps": 1, "alpha": 0.3,
                                        "steps": 10},
                                       ]
                    else:
                        atk_args_ls = [{"attack_eval": False},
                                       {"attack_eval": True, "method": "pgd", "eps": 8 / 255, "alpha": 2 / 255,
                                        "steps": 10},
                                       {"attack_eval": True, "method": "pgd", "eps": 2 / 255, "alpha": 0.5 / 255,
                                        "steps": 10},
                                       {"attack_eval": True, "method": "pgdl2", "eps": 0.2, "alpha": 0.05,
                                        "steps": 10},
                                       ]
                    for j, atk_args in enumerate(atk_args_ls):
                        accs = []
                        for it_eval in range(args.num_eval):
                            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, atk_args)
                            accs.append(acc_test)
                        print('Setting %d: evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(j, len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.



            ''' Train synthetic data '''
            #net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net = random.choice(network_pool)
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()

            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

            loss_avg = 0

            ''' update synthetic data '''
            if 'BN' not in args.model: # for ConvNet
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    if args.method == 'Attack':
                        seed = int(time.time() * 1000) % 100000
                        # We only want to attack the real dataset
                        img_real = Attack(img_real, lab_real, net, args.attack_strategy, seed=seed, param=args.attack_param)

                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)

                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

            else: # for ConvNetBN
                images_real_all = []
                images_syn_all = []
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    if args.method == 'Attack':
                        seed = int(time.time() * 1000) % 100000
                        img_real = Attack(img_real, lab_real, net, args.attack_strategy, seed=seed, param=args.attack_param)

                    images_real_all.append(img_real)
                    images_syn_all.append(img_syn)

                images_real_all = torch.cat(images_real_all, dim=0)
                images_syn_all = torch.cat(images_syn_all, dim=0)

                output_real = embed(images_real_all).detach()
                output_syn = embed(images_syn_all)

                loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)


            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()

            loss_avg /= (num_classes)

            if it%100 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))

            ''' update network '''
            for param in list(net.parameters()):
                param.requires_grad = True
            image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                label_syn.detach())  # avoid any unaware modification
            dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
            trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True,
                                                      num_workers=0)
            for il in range(args.inner_loop):
                epoch('train', trainloader, net, optimizer_net, criterion, args,
                      aug=True if args.dsa or args.attack or args.cure else False, adv=True)


            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()


