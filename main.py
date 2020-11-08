from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models import *
import torch.utils.tensorboard as tb
from utils import tee

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--tag', required=True)
parser.add_argument('--comment', required=True,
                    help='required comment describing the experiment (include blank string to skip)')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', required=True,
                    help='dir containing SceneFlow dataset')
parser.add_argument('--epochs', type=int, required=True,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', required=True,
                    help='output dir to save model into')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Echo inputs
with open(os.path.join(output_path, 'cmdline.sh'), mode='w') as f:
    print('#!/bin/bash', file=f)
    print('python main.py \\', file=f)
    print('    --tag', args.tag, '\\', file=f)
    print('    --comment "', args.comment, '" \\', file=f)
    print('    --maxdisp', args.maxdisp, '\\', file=f)
    print('    --model', args.model, '\\', file=f)
    print('    --datapath', args.datapath, '\\', file=f)
    print('    --epochs', args.epochs, '\\', file=f)
    if args.loadmodel is not None:
        print('    --loadmodel', args.loadmodel, '\\', file=f)
    print('    --savemodel', args.savemodel, '\\', file=f)
    print('    --no-cuda', (not args.cuda), '\\', file=f)
    print('    --seed', args.seed, '\\', file=f)
    print('', file=f)

with open(os.path.join(output_path, 'README.txt'), mode='w') as f:
    print(args.comment, file=f)
    if args.cuda:
        print('', file=f)
        print('CUDA has been requested', file=f)
        print('* CUDA is', ('not' if not torch.cuda.is_available() else ''), 'available', file=f)
        print('* Current device: ', torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
        print('* This is device #', torch.cuda.current_device(), 'of', torch.cuda.device_count(), file=f)
        print('', file=f)

# Prepare file system
output_path = os.path.join('/', 'cs230-datasets', 'PSMNet', args.tag)
assert not os.path.exists(output_path), "Cannot overwrite dir " + output_path
os.mkdir(output_path)

logfilepath = os.path.join(output_path, 'main.log')
logfile = open(logfilepath, mode='w')
assert logfile is not None
print('Writing logs to ', logfilepath)

savedir = os.path.join(output_path, args.savemodel)
os.mkdir(savedir)
print('Writing model into ', savedir)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)    

# Prepare dataset
all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(
    args.datapath)

train_batch_size = 8
TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
    batch_size=train_batch_size, shuffle=True, num_workers=8, drop_last=False)

test_batch_size = 8
TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
    batch_size=test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# Prepare model
if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    tee.tee('no model', file=logfile)

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    tee.tee('Load pretrained model', file=logfile)
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

tee.tee('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])), \
    file=logfile)

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

start_full_time = time.time()
tb_writer = tb.SummaryWriter(os.path.join('/', 'cs230-datasets', 'PSMNet', 'tensorboard', args.tag),
                             comment=args.comment)

def train(global_idx, imgL, imgR, disp_L):
    model.train()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

   # ---------
    mask = disp_true < args.maxdisp
    mask.detach_()
    # ----
    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + \
               0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + \
               1.0*F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(
            output[mask], disp_true[mask], size_average=True)
    
    tb_writer.add_scalar('train_loss', loss.item(), global_step=global_idx,
                         walltime=time.time() - start_full_time)

    loss.backward()
    optimizer.step()

    return loss.data


def test(batch_idx, imgL, imgR, disp_true):

    model.eval()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
    # ---------
    mask = disp_true < 192
    # ----

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16
        top_pad = (times+1)*16 - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 16 != 0:
        times = imgL.shape[3]//16
        right_pad = (times+1)*16-imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

    with torch.no_grad():
        output3 = model(imgL, imgR)
        output3 = torch.squeeze(output3)

    if top_pad != 0:
        img = output3[:, top_pad:, :]
    else:
        img = output3

    if len(disp_true[mask]) == 0:
        loss = 0
    else:
        # torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error
        loss = F.l1_loss(img[mask], disp_true[mask])

    loss_cpu = loss.data.cpu()
    tb_writer.add_scalar('test_loss', loss_cpu.item(), global_step=batch_idx,
                         walltime=time.time() - start_full_time)
    return loss_cpu


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    tee.tee('learning_rate = {}'.format(lr), file=logfile)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    for epoch in range(0, args.epochs):
        tee.tee('This is %d-th epoch' % (epoch), file=logfile)
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()

            global_idx = epoch * train_batch_size + batch_idx
            loss = train(global_idx, imgL_crop, imgR_crop, disp_crop_L)
            tee.tee('Iter %d training loss = %.3f , time = %.2f' %
                    (batch_idx, loss, time.time() - start_time), file=logfile)
            total_train_loss += loss
        tee.tee('epoch %d total training loss = %.3f' %
                (epoch, total_train_loss/len(TrainImgLoader)), file=logfile)

        # SAVE
        savefilename = os.path.join(savedir, 'checkpoint_'+str(epoch)+'.tar')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss/len(TrainImgLoader),
        }, savefilename)

    tee.tee('full training time = %.2f HR' %
            ((time.time() - start_full_time)/3600), file=logfile)

    # ------------- TEST ------------------------------------------------------------
    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        test_loss = test(batch_idx, imgL, imgR, disp_L)
        tee.tee('Iter %d test loss = %.3f' % (batch_idx, test_loss), file=logfile)
        total_test_loss += test_loss

    tee.tee('total test loss = %.3f' % (total_test_loss/len(TestImgLoader)), file=logfile)
    # ----------------------------------------------------------------------------------
    # SAVE test information
    savefilename = os.path.join(output_path, args.savemodel, 'testinformation.tar')
    torch.save({
        'test_loss': total_test_loss/len(TestImgLoader),
    }, savefilename)


if __name__ == '__main__':
    main()
