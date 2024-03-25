import torch
from tqdm import tqdm
import numpy as np
import torchvision
import argparse
from scipy import stats
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4"

def evaluate(device, model, data_loader):
    model.eval()
    losses = []
    acc_meter1 = AverageMeter()
    precision_meter1 = AverageMeter()
    recall_meter1 = AverageMeter()
    F1_meter1 = AverageMeter()
    IoU_meter1 = AverageMeter()   
    acc_meter2 = AverageMeter()
    precision_meter2 = AverageMeter()
    recall_meter2 = AverageMeter()
    F1_meter2 = AverageMeter()
    IoU_meter2 = AverageMeter()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            _, inputs, targets, boundary,_ = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs1,outputs2,outputs3 = model(inputs.float())
            output_mask = outputs1.sigmoid().detach().cpu().numpy().squeeze()
            output_boundary = outputs2[-1].detach().cpu().numpy().squeeze()
            targets1 = targets.detach().cpu().numpy().squeeze()
            targets2 = boundary.detach().cpu().numpy().squeeze()
            res1 = np.zeros((256, 256))
            res2 = np.zeros((256, 256))
            res1[output_mask > 0.5] = 255
            res1[output_mask <=0.5] = 0
            res2[output_boundary > 0.5] = 255
            res2[output_boundary <=0.5] = 0
            acc_r, precision_r, recall_r, F1_r, IoU_r = binary_accuracy(res1, targets1)
            acc_b, precision_b, recall_b, F1_b, IoU_b = binary_accuracy(res2, targets2)
            acc_meter1.update(acc_r)
            '''
            precision_meter1.update(precision_r)
            recall_meter1.update(recall_r)
            F1_meter1.update(F1_r)
            IoU_meter1.update(IoU_r)            
            acc_meter2.update(acc_b)
            precision_meter2.update(precision_b)
            recall_meter2.update(recall_b)
            F1_meter2.update(F1_b)
            IoU_meter2.update(IoU_b)
            '''
    return acc_meter1.avg * 100


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def align_dims(np_input, expected_dims=2):
    dim_input = len(np_input.shape)
    np_output = np_input
    if dim_input>expected_dims:
        np_output = np_input.squeeze(0)
    elif dim_input<expected_dims:
        np_output = np.expand_dims(np_input, 0)
    assert len(np_output.shape) == expected_dims
    return np_output

def binary_accuracy(pred, label):
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)
    pred = (pred >= 0.5)
    label = (label >= 0.5)

    TP = float((pred * label).sum())
    FP = float((pred * (1 - label)).sum())
    FN = float(((1 - pred) * (label)).sum())
    TN = float(((1 - pred) * (1 - label)).sum())
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    IoU = TP / (TP + FP + FN + 1e-10)
    acc = (TP + TN) / (TP + FP + FN + TN)
    F1 = 0
    if acc > 0.99 and TP == 0:
        precision = 1
        recall = 1
        IoU = 1
    if precision > 0 and recall > 0:
        F1 = stats.hmean([precision, recall])
    return acc, precision, recall, F1, IoU


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()
