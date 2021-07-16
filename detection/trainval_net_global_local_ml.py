# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import os
import shutil
import logging
import pprint
import time
import numpy as np
import _init_paths


import torch
import torch.nn as nn
from torch.autograd import Variable

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, FocalLoss, sampler, calc_supp, EFocalLoss
from model.utils.parser_func import parse_args, set_dataset_args

from da_utils.utils import create_logger, get_parameters
from model.faster_rcnn.vgg16_global_local import vgg16
from model.faster_rcnn.resnet_global_local import resnet


if __name__ == '__main__':
    args = parse_args()
    args = set_dataset_args(args)
    output_dir = os.path.join(args.save_dir)
    if not os.path.exists(os.path.join(output_dir, 'ckpt')):
        os.makedirs(os.path.join(output_dir, 'ckpt'), exist_ok=True)
    logger_ = create_logger(output_dir)

    logging.info('====================== args ============================')
    logging.info(pprint.pformat(args))

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    logging.info('====================== config ============================')
    logging.info(pprint.pformat(cfg))
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        logging.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
    train_size_t = len(roidb_t)
    logging.info("==> src: {} | tar: {}".format(len(roidb), len(roidb_t)))

    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)
    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True)
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                               sampler=sampler_batch, num_workers=args.num_workers)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, imdb.num_classes, training=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
                                               sampler=sampler_batch_t, num_workers=args.num_workers)
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    if args.cuda:
        cfg.CUDA = True

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, lc=args.lc, gc=args.gc)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic, lc=args.lc, gc=args.gc)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic, context=args.context)
    fasterRCNN.create_architecture()

    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    if args.cuda:
        fasterRCNN.cuda()

    ckpt_last = os.path.join(output_dir, 'ckpt', 'model.pth.tar-last')
    if os.path.isfile(ckpt_last):
        checkpoint = torch.load(ckpt_last)
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        logging.info("==> loaded checkpoint from {} | epoch: {}".format(args.load_name, args.start_epoch))
    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    iters_per_epoch = int(10000 / args.batch_size)
    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)
    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")
    count_iter = 0
    logging.info('\n\n====================== running meta lr============================')
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter_s = iter(dataloader_s)
        data_iter_t = iter(dataloader_t)
        meta_flags = False
        for step in range(iters_per_epoch):
            if step % args.step == 0:
                meta_flags = not meta_flags
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(dataloader_s)
                data_s = next(data_iter_s)
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(dataloader_t)
                data_t = next(data_iter_t)
            count_iter += 1

            # ================= detection first =======================
            if meta_flags:
                fasterRCNN.zero_grad()
                for p in fasterRCNN.parameters():
                    p.fast = None

                im_data.data.resize_(data_s[0].size()).copy_(data_s[0])
                im_info.data.resize_(data_s[1].size()).copy_(data_s[1])
                gt_boxes.data.resize_(data_s[2].size()).copy_(data_s[2])
                num_boxes.data.resize_(data_s[3].size()).copy_(data_s[3])

                _, _, _, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label, _, _ = \
                    fasterRCNN(im_data, im_info, gt_boxes, num_boxes, batch_size=data_s[0].size(0), only_det=True)
                loss_det = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

                # meta_grad
                meta_grad = torch.autograd.grad(
                    outputs=loss_det, inputs=get_parameters(fasterRCNN), create_graph=True, allow_unused=True)
                meta_grad = [g if g is None else g.detach() for g in meta_grad]
                for k, param in enumerate(get_parameters(fasterRCNN)):
                    if meta_grad[k] is not None:
                        param.fast = param - args.meta_lr * lr * meta_grad[k]

                # meta test
                out_d_pixel, out_d = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                                batch_size=data_s[0].size(0), only_domain=True)
                domain_s = Variable(torch.zeros(out_d.size(0)).long().cuda())
                dloss_s = 0.5 * FL(out_d, domain_s)
                dloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)

                im_data.data.resize_(data_t[0].size()).copy_(data_t[0])
                im_info.data.resize_(data_t[1].size()).copy_(data_t[1])
                gt_boxes.data.resize_(1, 1, 5).zero_()
                num_boxes.data.resize_(1).zero_()
                out_d_pixel, out_d = fasterRCNN(
                    im_data, im_info, gt_boxes, num_boxes, target=True, batch_size=data_t[0].size(0), only_domain=True)
                domain_t = Variable(torch.ones(out_d.size(0)).long().cuda())
                dloss_t = 0.5 * FL(out_d, domain_t)
                dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)

                loss = loss_det + dloss_t + dloss_s + dloss_t_p + dloss_s_p
                loss_temp += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                fasterRCNN.zero_grad()
                for p in fasterRCNN.parameters():
                    p.fast = None

                # domain first
                im_data.data.resize_(data_s[0].size()).copy_(data_s[0])
                im_info.data.resize_(data_s[1].size()).copy_(data_s[1])
                gt_boxes.data.resize_(data_s[2].size()).copy_(data_s[2])
                num_boxes.data.resize_(data_s[3].size()).copy_(data_s[3])
                out_d_pixel, out_d = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                                batch_size=data_s[0].size(0), only_domain=True)
                domain_s = Variable(torch.zeros(out_d.size(0)).long().cuda())
                dloss_s = 0.5 * FL(out_d, domain_s)
                dloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)
                im_data.data.resize_(data_t[0].size()).copy_(data_t[0])
                im_info.data.resize_(data_t[1].size()).copy_(data_t[1])
                gt_boxes.data.resize_(1, 1, 5).zero_()
                num_boxes.data.resize_(1).zero_()
                out_d_pixel, out_d = fasterRCNN(
                    im_data, im_info, gt_boxes, num_boxes, target=True, batch_size=data_t[0].size(0), only_domain=True)
                domain_t = Variable(torch.ones(out_d.size(0)).long().cuda())
                dloss_t = 0.5 * FL(out_d, domain_t)
                dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)
                loss_d = dloss_t + dloss_s + dloss_t_p + dloss_s_p

                # meta grad
                meta_grad = torch.autograd.grad(
                    outputs=loss_d, inputs=get_parameters(fasterRCNN), create_graph=True, allow_unused=True
                )
                meta_grad = [g if g is None else g.detach() for g in meta_grad]
                for k, param in enumerate(get_parameters(fasterRCNN)):
                    if meta_grad[k] is not None:
                        param.fast = param - args.meta_lr * lr * meta_grad[k]

                # meta test
                im_data.data.resize_(data_s[0].size()).copy_(data_s[0])
                im_info.data.resize_(data_s[1].size()).copy_(data_s[1])
                gt_boxes.data.resize_(data_s[2].size()).copy_(data_s[2])
                num_boxes.data.resize_(data_s[3].size()).copy_(data_s[3])
                _, _, _, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label, _, _ = \
                    fasterRCNN(im_data, im_info, gt_boxes, num_boxes, batch_size=data_s[0].size(0), only_det=True)
                loss_det = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

                loss = loss_d + loss_det
                loss_temp += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)
                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    dloss_s = dloss_s.item()
                    dloss_t = dloss_t.item()
                    dloss_s_p = dloss_s_p.item()
                    dloss_t_p = dloss_t_p.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                logging.info("E: {:>3d}/{:>3d} | I: {:>5d}/{:>5d} | f/b: {:0>3d}/{:0>3d} | t: {:.1f} | loss: {:.4f} | "
                             "lr: {:.2e} | rpn: {:.4f}/{:.4f} | rcnn: {:.4f}/{:.4f} | D: {:.4f}/{:.4f} | "
                             "D_p: {:.4f}/{:.4f}".format(
                    epoch, args.max_epochs, step, iters_per_epoch, fg_cnt, bg_cnt, end - start, loss_temp, lr,
                    loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, dloss_s, dloss_t, dloss_s_p, dloss_t_p))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)
                loss_temp = 0
                start = time.time()
        save_name_epoch = os.path.join(output_dir, 'ckpt', 'model.pth.tar-{}'.format(epoch))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name_epoch)
        save_name_last = os.path.join(output_dir, 'ckpt', 'model.pth.tar-last')
        shutil.copyfile(save_name_epoch, save_name_last)

        logging.info('==> saved model: {}'.format(save_name_epoch))

    if args.use_tfboard:
        logger.close()
