import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, Attack, ParamAttack


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--attack_strategy', type=str, default='pgd', help='attack strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.attack_param = ParamAttack()
    args.dsa = True if args.method == 'DSA' else False
    args.attack = True if args.method == "Attack" else False
    args.attack_eval = False

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    norm_accs_all_exps = dict() # record performances of all experiments
    atk_accs_all_exps = dict() 
    for key in model_eval_pool:
        norm_accs_all_exps[key] = []
        atk_accs_all_exps[key] = []

    norm_data_save = []
    atk_data_save = []


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
        norm_image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        norm_label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        atk_image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        atk_label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]


        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                images = get_images(c, args.ipc).detach().data
                norm_image_syn.data[c*args.ipc:(c+1)*args.ipc] = copy.deepcopy(images)
                atk_image_syn.data[c*args.ipc:(c+1)*args.ipc] = copy.deepcopy(images)
        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        norm_optimizer_img = torch.optim.SGD([norm_image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        norm_optimizer_img.zero_grad()
        atk_optimizer_img = torch.optim.SGD([atk_image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        atk_optimizer_img.zero_grad()

        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    # elif args.attack:
                    #     args.epoch_eval_train = 1000
                    #     args.dc_aug_param = None
                    #     print('Attack strategy: \n', args.attack_strategy)
                    #     print('Attack parameters: \n', args.attack_param.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dsa or args.attack or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300

                    norm_accs = []
                    atk_accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

                        norm_image_syn_eval, norm_label_syn_eval = copy.deepcopy(norm_image_syn.detach()), copy.deepcopy(norm_label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, norm_image_syn_eval, norm_label_syn_eval, testloader, args)
                        norm_accs.append(acc_test)

                        atk_image_syn_eval, atk_label_syn_eval = copy.deepcopy(atk_image_syn.detach()), copy.deepcopy(atk_label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, atk_image_syn_eval, atk_label_syn_eval, testloader, args)
                        atk_accs.append(acc_test)

                    print('(N) Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(norm_accs), model_eval, np.mean(norm_accs), np.std(norm_accs)))
                    print('(A) Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(atk_accs), model_eval, np.mean(atk_accs), np.std(atk_accs)))
                    if it == args.Iteration: # record the final results
                        norm_accs_all_exps[model_eval] += norm_accs
                        atk_accs_all_exps[model_eval] += atk_accs

                ''' visualize and save '''
                norm_save_name = os.path.join(args.save_path, 'vis_alp_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                norm_image_syn_vis = copy.deepcopy(norm_image_syn.detach().cpu())

                atk_save_name = os.path.join(args.save_path, 'vis_alp_%s_%s_%s_%dipc_exp%d_iter%d.png'%('Attack', args.dataset, args.model, args.ipc, exp, it))
                atk_image_syn_vis = copy.deepcopy(atk_image_syn.detach().cpu())
                
                for ch in range(channel):
                    norm_image_syn_vis[:, ch] = norm_image_syn_vis[:, ch]  * std[ch] + mean[ch]
                    atk_image_syn_vis[:, ch] = atk_image_syn_vis[:, ch]  * std[ch] + mean[ch]
                
                norm_image_syn_vis[norm_image_syn_vis<0] = 0.0
                norm_image_syn_vis[norm_image_syn_vis>1] = 1.0
                save_image(norm_image_syn_vis, norm_save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

                atk_image_syn_vis[atk_image_syn_vis<0] = 0.0
                atk_image_syn_vis[atk_image_syn_vis>1] = 1.0
                save_image(atk_image_syn_vis, atk_save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.



            ''' Train synthetic data '''
            norm_net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            norm_net.train()
            norm_net_parameters = list(norm_net.parameters())
            norm_optimizer_net = torch.optim.SGD(norm_net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            norm_optimizer_net.zero_grad()

            atk_net = copy.deepcopy(norm_net)
            atk_net.train()
            atk_net_parameters = list(atk_net.parameters())
            atk_optimizer_net = torch.optim.SGD(atk_net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            atk_optimizer_net.zero_grad()

            norm_loss_avg = 0.0  
            atk_loss_avg = 0.0
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.


            for ol in range(args.outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in norm_net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True

                for module in atk_net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    norm_net.train() # for updating the mu, sigma of BatchNorm
                    output_real = norm_net(img_real) # get running mu, sigma
                    for module in norm_net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer

                    atk_net.train() # for updating the mu, sigma of BatchNorm
                    output_real = atk_net(img_real) # get running mu, sigma
                    for module in atk_net.modules():
                        if 'BatchNorm' in module._get_name():
                            module.eval()


                ''' update synthetic data '''
                norm_loss = torch.tensor(0.0).to(args.device)
                atk_loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c

                    norm_img_syn = norm_image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    norm_lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c
                    atk_img_syn = atk_image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    atk_lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    if args.method == 'DSA':
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        norm_img_syn = DiffAugment(norm_img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        atk_img_syn = DiffAugment(norm_img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    # if args.method == 'Attack':
                    #     seed = int(time.time() * 1000) % 100000
                    #     # We only want to attack the real dataset
                    #     img_real = Attack(img_real, lab_real, net, args.attack_strategy, seed=seed, param=args.attack_param)

                    output_real = norm_net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, norm_net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    norm_output_syn = norm_net(norm_img_syn)
                    norm_loss_syn = criterion(norm_output_syn, norm_lab_syn)
                    norm_gw_syn = torch.autograd.grad(norm_loss_syn, norm_net_parameters, create_graph=True)
                    norm_loss += match_loss(norm_gw_syn, gw_real, args)

                    # After finishing calculating the gradient of the real data, we can attack the synthetic data
                    seed = int(time.time() * 1000) % 100000
                    img_real = Attack(img_real, lab_real, atk_net, args.attack_strategy, seed=seed, param=args.attack_param)

                    output_real = atk_net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, atk_net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    atk_output_syn = atk_net(atk_img_syn)
                    atk_loss_syn = criterion(atk_output_syn, atk_lab_syn)
                    atk_gw_syn = torch.autograd.grad(atk_loss_syn, atk_net_parameters, create_graph=True)
                    atk_loss += match_loss(atk_gw_syn, gw_real, args)

                norm_optimizer_img.zero_grad()
                norm_loss.backward()
                norm_optimizer_img.step()
                norm_loss_avg += norm_loss.item()

                atk_optimizer_img.zero_grad()
                atk_loss.backward()
                atk_optimizer_img.step()
                atk_loss_avg += atk_loss.item()

                if ol == args.outer_loop - 1:
                    break

                ''' update network '''
                norm_image_syn_train, norm_label_syn_train = copy.deepcopy(norm_image_syn.detach()), copy.deepcopy(norm_label_syn.detach())  # avoid any unaware modification
                norm_dst_syn_train = TensorDataset(norm_image_syn_train, norm_label_syn_train)
                norm_trainloader = torch.utils.data.DataLoader(norm_dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

                atk_image_syn_train, atk_label_syn_train = copy.deepcopy(atk_image_syn.detach()), copy.deepcopy(atk_label_syn.detach())  # avoid any unaware modification
                atk_dst_syn_train = TensorDataset(atk_image_syn_train, atk_label_syn_train)
                atk_trainloader = torch.utils.data.DataLoader(atk_dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

                for il in range(args.inner_loop):
                    epoch('train', norm_trainloader, norm_net, norm_optimizer_net, criterion, args, aug = True if args.dsa or args.attack else False)
                    epoch('train', atk_trainloader, atk_net, atk_optimizer_net, criterion, args, aug = True if args.dsa or args.attack else False)

            norm_loss_avg /= (num_classes*args.outer_loop)
            atk_loss_avg /= (num_classes*args.outer_loop)

            if it%10 == 0:
                print('%s iter = %04d, norm loss = %.4f' % (get_time(), it, norm_loss_avg))
                print('%s iter = %04d, atk loss = %.4f' % (get_time(), it, atk_loss_avg))


            if it == args.Iteration: # only record the final results
                norm_data_save.append([copy.deepcopy(norm_image_syn.detach().cpu()), copy.deepcopy(norm_label_syn.detach().cpu())])
                torch.save({'data': norm_data_save, 'accs_all_exps': norm_accs_all_exps, }, os.path.join(args.save_path, 'res_alp_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))
                atk_data_save.append([copy.deepcopy(atk_image_syn.detach().cpu()), copy.deepcopy(atk_label_syn.detach().cpu())])
                torch.save({'data': atk_data_save, 'accs_all_exps': atk_accs_all_exps, }, os.path.join(args.save_path, 'res_alp_%s_%s_%s_%dipc.pt'%('Attack', args.dataset, args.model, args.ipc)))

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        norm_accs = norm_accs_all_exps[key]
        print('Run %d experiments on norm, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(norm_accs), key, np.mean(norm_accs)*100, np.std(norm_accs)*100))

        atk_accs = atk_accs_all_exps[key]
        print('Run %d experiments on atk, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(atk_accs), key, np.mean(atk_accs)*100, np.std(atk_accs)*100))


if __name__ == '__main__':
    main()


