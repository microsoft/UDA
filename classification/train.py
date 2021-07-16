# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import os
import random
import logging
import argparse
import os.path as osp
from pprint import pformat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import loss as loss
from pre_process import image_target, image_test

from lr_scheduler import inv_lr_scheduler, get_current_lr
from resnet import resnet_advnet
from data_list import ImageList
from utils import create_logger, write_log, save_model, AverageMeter, calc_coeff
from torch_ops import get_parameters, image_classification_test


def train(config):
    dset_loaders = {}
    data_config = config["data"]

    # prepare data
    logging.info("==> loading dataset")
    dset_loaders["source"] = DataLoader(
        ImageList(open(data_config["source"]["list_path"]).readlines(), transform=image_target()),
        batch_size=data_config["source"]["batch_size"], shuffle=True, num_workers=8, drop_last=True)
    dset_loaders["target"] = DataLoader(
        ImageList(open(data_config["target"]["list_path"]).readlines(), transform=image_target()),
        batch_size=data_config["source"]["batch_size"], shuffle=True, num_workers=8, drop_last=True)
    dset_loaders["test"] = DataLoader(
        ImageList(open(data_config["test"]["list_path"]).readlines(), transform=image_test()),
        batch_size=data_config["test"]["batch_size"], shuffle=False, num_workers=8)

    # set network
    logging.info("==> building network")
    class_num = config["network"]["params"]["class_num"]
    base_network, ad_net = resnet_advnet(
        num_classes=class_num, in_dim=config['adnet_indim'], gvb=config['gvb'])
    base_network = base_network.cuda()
    ad_net = ad_net.cuda()

    # set weightnet
    # if config['stage']:
    #     logging.info("==> building weightnet from WeightParams")
    #     weightnet = WeightParams(num_classes=4).cuda()

    # set optimizer
    logging.info("==> setting optimizer")
    # if config['stage']:
    #     parameter_list = base_network.get_parameters() + ad_net.get_parameters() + weightnet.get_parameters()
    # else:
    #     parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()

    optimizer_config = config["optimizer"]
    optimizer = optim.SGD(parameter_list, **(optimizer_config["optim_params"]))
    schedule_param = optimizer_config["lr_param"]

    # load ckpt
    losses_cls, losses_trs, losses_gvbg, losses_gvbd = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    start_ite, best_acc, mean_ent = 0, 0., 0.
    last_ckpt = os.path.join(config['output_path'], 'model.pth.tar-last')
    if os.path.exists(last_ckpt):
        ckpt = torch.load(last_ckpt)
        base_ckpt = ckpt['base_dict']
        base_network.load_state_dict(base_ckpt)
        adv_dict = ckpt['adv_dict']
        ad_net.load_state_dict(adv_dict)
        opt_dict = ckpt['optimizer']
        optimizer.load_state_dict(opt_dict)
        start_ite = ckpt['ite']
        best_acc = ckpt['best_acc']
        # if 'weight_dict' in ckpt.keys():
        #     weightnet.load_state_dict(ckpt['weight_dict'])
        #     logging.info("==> loading ckpt from {} with weightnet | ite: {} | best_acc: {:.3f}".format(
        #         last_ckpt, start_ite, best_acc * 100.))
        logging.info("==> loading ckpt from {} | ite: {} | best_acc: {:.3f}".format(
            last_ckpt, start_ite, best_acc * 100.))

    # train
    logging.info("\n===============> starting training <==================")
    loss_params = config["loss"]
    len_train_src, len_train_tar = len(dset_loaders["source"]), len(dset_loaders["target"])
    step_flags = False
    for i in range(start_ite, config["num_iterations"]):
        if i % config['step'] == 0:
            step_flags = not step_flags
        # test
        if i % config["test_interval"] == config["test_interval"] - 1 and i != 0:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, base_network, gvb=config['gvb'])
            is_best = False
            if temp_acc > best_acc:
                best_acc = temp_acc
                is_best = True
            logging.info("==> testing on {}".format(data_config["test"]["list_path"]))
            log_str = "I: {}/{} | acc: {:.3f} | best acc: {:.3f}".format(
                i, config['num_iterations'], temp_acc * 100, best_acc * 100)
            write_log(config["out_file"], log_str)
            logging.info(log_str)
            data_dict = {
                "base_dict": base_network.state_dict(),
                "adv_dict": ad_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ite": i,
                "best_acc": best_acc
            }
            # if config['stage']:
            #     data_dict.update({"weight_dict": weightnet.state_dict()})
            save_model(config["output_path"], data_dict=data_dict, ite=i, is_best=is_best)

        # train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer = inv_lr_scheduler(optimizer, i, **schedule_param)
        current_lr = get_current_lr(optimizer)

        # dataloader
        if i % len_train_src == 0 or i == start_ite:
            iter_src = iter(dset_loaders["source"])
        if i % len_train_tar == 0 or i == start_ite:
            iter_tar = iter(dset_loaders["target"])

        if step_flags:
            # =================== cls first =======================
            for p in base_network.parameters():
                p.fast = None

            # classification on source
            data_src, data_tar = iter_src.__next__(), iter_tar.__next__()
            inputs_src, labels_src = data_src[0].cuda(), data_src[1].cuda()
            outputs_src_y, outputs_src_z = base_network(inputs_src, gvbg=config['gvb'])
            outputs_src = outputs_src_y - outputs_src_z if config['gvb'] else outputs_src_y

            loss_cls = F.cross_entropy(outputs_src, labels_src)
            if config['gvb']:
                loss_gvbg = torch.mean(torch.abs(outputs_src_z.reshape(-1)))
                losses_gvbg.update(loss_gvbg.item(), outputs_src_z.size(0))
                meta_grad = torch.autograd.grad(outputs=loss_cls + loss_gvbg,
                                                inputs=base_network.parameters(), create_graph=True)
            else:
                meta_grad = torch.autograd.grad(outputs=loss_cls, inputs=base_network.parameters(),
                                                create_graph=True)
            meta_grad = [g.detach() for g in meta_grad]
            for k, param in enumerate(base_network.named_parameters()):
                if meta_grad[k] is not None:
                    _lr = 10. if 'fc' in param[0] or 'gvb' in param[0] else 1.
                    param[1].fast = param[1] - config['meta_lr'] * current_lr * _lr * meta_grad[k]

            # meta test
            inputs_src, inputs_tar, labels_src = data_src[0].cuda(), data_tar[0].cuda(), data_src[1].cuda()
            _batch_size = inputs_src.size(0)

            if config['split']:
                features_src, outputs_src_y, outputs_src_z = base_network(inputs_src, True, gvbg=config['gvb'])
                features_tar, outputs_tar_y, outputs_tar_z = base_network(inputs_tar, True, gvbg=config['gvb'])
                features = torch.cat((features_src, features_tar), dim=0)
                outputs_z = torch.cat((outputs_src_z, outputs_tar_z), dim=0)
            else:
                inputs = torch.cat((inputs_src, inputs_tar), dim=0)
                features, outputs_y, outputs_z = base_network(inputs, True, gvbg=config['gvb'])
                outputs_src_y, outputs_tar_y = torch.split(outputs_y, _batch_size)
                outputs_src_z, outputs_tar_z = torch.split(outputs_z, _batch_size)
            if config['gvb']:
                outputs_src = outputs_src_y - outputs_src_z
                outputs_tar = outputs_tar_y - outputs_tar_z
            else:
                outputs_src = outputs_src_y
                outputs_tar = outputs_tar_y
            outputs = torch.cat((outputs_src, outputs_tar), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)

            # loss calculation
            if config['method'] == 'GVB':
                loss_trs, mean_ent, loss_gvbd = loss.DA(softmax_out, ad_net, coeff=calc_coeff(i), gvb=True)
                if config['gvbg']:
                    loss_gvbg_test = torch.mean(torch.abs(outputs_z))
                    loss_gvb = loss_gvbd + (loss_gvbg + loss_gvbg_test) * 0.5
                else:
                    loss_gvb = loss_gvbd + loss_gvbg
                total_loss = loss_cls + loss_trs * loss_params['trade_off'] + loss_gvb
                losses_gvbd.update(loss_gvbd.item(), _batch_size * 2)
            losses_trs.update(loss_trs.item(), outputs_z.size(0))
            losses_cls.update(loss_cls.item(), outputs_z.size(0) // 2)

            if i % config["print_num"] == 0:
                log_str = "I: {}/{} | lr: {:.5f} | L_trs: {:.4f} | gvb: {:.5f}/{:.5f} | L_cls: {:.4f} | " \
                          "MEnt: {:.4f} | best_acc: {:.3f}".format(
                    i, config['num_iterations'], current_lr, losses_trs.avg, losses_gvbg.avg, losses_gvbd.avg,
                    losses_cls.avg, mean_ent, best_acc * 100, current_lr
                )
                logging.info(log_str)
                write_log(config["out_file"], log_str)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        else:
            # =================== trs first =======================
            for p in base_network.parameters():
                p.fast = None
            for p in ad_net.parameters():
                p.fast = None

            # meta train
            data_src, data_tar = iter_src.__next__(), iter_tar.__next__()
            inputs_src, labels_src = data_src[0].cuda(), data_src[1].cuda()
            inputs_tar = data_tar[0].cuda()
            _batch_size = inputs_src.size(0)

            if config['split']:
                features_src, outputs_src_y, outputs_src_z = base_network(inputs_src, True, gvbg=config['gvb'])
                features_tar, outputs_tar_y, outputs_tar_z = base_network(inputs_tar, True, gvbg=config['gvb'])
                features = torch.cat((features_src, features_tar), dim=0)
                outputs_z = torch.cat((outputs_src_z, outputs_tar_z), dim=0)
            else:
                inputs = torch.cat((inputs_src, inputs_tar), dim=0)
                features, outputs_y, outputs_z = base_network(inputs, True, gvbg=config['gvb'])
                outputs_src_y, outputs_tar_y = torch.split(outputs_y, _batch_size)
                outputs_src_z, outputs_tar_z = torch.split(outputs_z, _batch_size)
            if config['gvb']:
                outputs_src = outputs_src_y - outputs_src_z
                outputs_tar = outputs_tar_y - outputs_tar_z
            else:
                outputs_src = outputs_src_y
                outputs_tar = outputs_tar_y
            outputs = torch.cat((outputs_src, outputs_tar), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)

            #
            if config['method'] == 'GVB':
                loss_trs, mean_ent, loss_gvbd = loss.DA(softmax_out, ad_net, coeff=calc_coeff(i), gvb=True)
                if config['gvbg']:
                    loss_gvbd += torch.mean(torch.abs(outputs_z))
                loss_train = loss_trs * loss_params['trade_off'] + loss_gvbd
                losses_gvbd.update(loss_gvbd.item(), _batch_size * 2)

            losses_trs.update(loss_trs.item(), outputs_z.size(0))

            meta_grad = torch.autograd.grad(
                outputs=loss_train,
                inputs=get_parameters(base_network, ad_net),
                create_graph=True,
                allow_unused=True
            )
            meta_grad = [g.detach() if g is not None else g for g in meta_grad]
            for k, param in enumerate(get_parameters(base_network, ad_net, name=True)):
                if meta_grad[k] is not None:
                    _lr = 10. if 'fc' in param[0] or 'gvb' in param[0] else 1.
                    param[1].fast = param[1] - config['meta_lr'] * current_lr * _lr * meta_grad[k]

            # meta test
            inputs_src, labels_src = data_src[0].cuda(), data_src[1].cuda()
            outputs_src_y, outputs_src_z = base_network(inputs_src, gvbg=config['gvb'])
            outputs_src = outputs_src_y - outputs_src_z if config['gvb'] else outputs_src_y
            loss_cls = F.cross_entropy(outputs_src, labels_src)
            if config['gvb']:
                loss_gvbg_test = torch.mean(torch.abs(outputs_src_z.reshape(-1)))
                total_loss = loss_train + loss_cls + loss_gvbg_test
            else:
                total_loss = loss_train + loss_cls
            losses_cls.update(loss_cls.item(), outputs_src.size(0))
            # if config['stage']:
            #     total_loss += loss_weights * 0.01

            if i % config["print_num"] == 0:
                log_str = "I: {}/{} | lr: {:.5f} | L_trs: {:.4f} | gvb: {:.5f}/{:.5f} | L_cls: {:.4f} | " \
                          "MEnt: {:.4f} | best_acc: {:.3f}".format(
                    i, config['num_iterations'], current_lr, losses_trs.avg, losses_gvbg.avg, losses_gvbd.avg,
                    losses_cls.avg, mean_ent, best_acc * 100, current_lr
                )
                logging.info(log_str)
                write_log(config["out_file"], log_str)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # save model
        if i % config["snapshot_interval"] == 0 and i != 0:
            data_dict = {
                "base_dict": base_network.state_dict(),
                "adv_dict": ad_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ite": i,
                "best_acc": best_acc
            }
            save_model(config["output_path"], data_dict=data_dict, ite=i, snap=True, is_best=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MetaAlign')

    parser.add_argument('--seed',               type=int, default=0)

    parser.add_argument('--batch_size',         type=int, default=24, help="batch size")
    parser.add_argument('--test_interval',      type=int, default=500)
    parser.add_argument('--snapshot_interval',  type=int, default=5000)
    parser.add_argument('--print_num',          type=int, default=50)
    parser.add_argument('--num_iterations',     type=int, default=10002)
    parser.add_argument('--lr',                 type=float, default=0.001)
    parser.add_argument('--meta_lr',            type=float, default=1.)
    parser.add_argument('--trade_off',          type=float, default=1, help="parameter for transfer loss")
    parser.add_argument('--net',                type=str, default='ResNet50')

    parser.add_argument('--dset',               type=str, default='office')
    parser.add_argument('--output_dir',         type=str, default='')
    parser.add_argument('--output_root',        type=str, default='exp')
    parser.add_argument('--s_dset_path',        type=str, default='data/Art.txt')
    parser.add_argument('--t_dset_path',        type=str, default='data/Clipart.txt')

    parser.add_argument('--thres',              type=float, default=0.9)
    parser.add_argument('--warm_up',            type=int, default=10)
    parser.add_argument('--pseudo',             dest='pseudo', action='store_true')
    parser.add_argument('--method',             type=str, default='GVB', choices=['GVB'])
    parser.add_argument('--stage',              action='store_true', dest='stage')
    parser.add_argument('--split',              action='store_true', dest='split')
    parser.add_argument('--stage_sum',          type=float, default=1.)
    parser.add_argument('--stage_loss',         type=float, default=0.01)
    parser.add_argument('--step',               type=int, default=1)
    parser.add_argument('--gvbg',               action='store_true')
    parser.add_argument('--original',           action='store_true')

    args = parser.parse_args()

    # train config
    config = {}

    config["thres"] = args.thres
    config["warm_up"] = args.warm_up
    config['pseudo'] = args.pseudo
    config['method'] = args.method
    config['stage'] = args.stage
    config['split'] = args.split
    config['stage_sum'] = args.stage_sum
    config['stage_loss'] = args.stage_loss
    config['step'] = args.step
    config['gvbg'] = args.gvbg
    config['original'] = args.original

    # ====================== running ===================
    config["num_iterations"] = args.num_iterations
    config["print_num"] = args.print_num
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval

    config["meta_lr"] = args.meta_lr

    config["prep"] = {
        'params': {
            "resize_size": 256,
            "crop_size": 224,
            'alexnet': False
        }
    }

    # ====================== directory ===================
    config["output_path"] = os.path.join(args.output_root, args.dset, args.output_dir)
    if not osp.exists(config["output_path"]):
        os.makedirs(config['output_path'])
    config["out_file"] = osp.join(config["output_path"], "log.txt")

    # ====================== model ===================
    config["optimizer"] = {
        "type": optim.SGD,
        "optim_params": {
            'lr': args.lr,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "nesterov": True
        },
        "lr_type": "inv",
        "lr_param": {
            "lr": args.lr,
            "gamma": 0.001,
            "power": 0.75
        }
    }
    config["loss"] = {
        "trade_off": args.trade_off
    }
    if "ResNet" in args.net:
        config["network"] = {
            "name": 'resnet50',
            "params": {
                "resnet_name": args.net,
                "use_bottleneck": False,
                "bottleneck_dim": 256,
                "new_cls": True
            }
        }
    else:
        raise ValueError('Network cannot be recognized. Please define your own dataset here.')

    # ====================== dataset ===================
    config["dataset"] = args.dset
    config["data"] = {
        "source": {
            "list_path": args.s_dset_path,
            "batch_size": args.batch_size
        },
        "target": {
            "list_path": args.t_dset_path,
            "batch_size": args.batch_size
        },
        "test": {
            "list_path": args.t_dset_path,
            "batch_size": args.batch_size
        }
    }
    if config["dataset"] == "office-home":
        config["network"]["params"]["class_num"] = 65
    elif config["dataset"] == "office":
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "visda":
        config["network"]["params"]["class_num"] = 12
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')

    # model
    if config['method'] in ['GVB']:
        in_f = config["network"]["params"]["class_num"]
    else:
        logging.info("==> Not implementation")
    config['adnet_indim'] = in_f
    config['gvb'] = True if 'GVB' in config['method'] else False
    config['seed'] = args.seed

    # set seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    write_log(config["out_file"], str(pformat(config)))
    logger = create_logger(config['output_path'])

    logger.info("================== configs ======================")
    logging.info(pformat(config))
    train(config)
