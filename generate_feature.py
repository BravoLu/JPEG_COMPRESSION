from __future__ import print_function
import argparse
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import numpy as np
from models import *
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_dir', type=str, default='./train',
                    help="the path to save the trained model")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='../data', train=True, transform=transform_test, download=True),
    batch_size=128, shuffle=False, **kwargs)

model = ResNet18().cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load('checkpoint/cifar10.pth')['net'])

if args.cuda:
    model.cuda()

def generate_feature():
    model.eval()
    cnt = 0
    out_target = []
    out_data = []
    out_output =[]
    out_advoutput = []
    out_advdata = []
    for data, target in train_loader:
        cnt += len(data)
        print("processing: %d/%d" % (cnt, len(train_loader.dataset)))
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, _ = model(data)
        output /= torch.norm(output, dim=1).unsqueeze(1)

        output_np = output.data.cpu().numpy()
        target_np = target.data.cpu().numpy()
        data_np = data.data.cpu().numpy()

        out_output.append(output_np)
        out_target.append(target_np[:, np.newaxis])
        out_data.append(np.squeeze(data_np))


    output_array = np.concatenate(out_output, axis=0)
    target_array = np.concatenate(out_target, axis=0)
    data_array = np.concatenate(out_data, axis=0)
    np.save(os.path.join(args.save_dir, 'output.npy'), output_array[:40000], allow_pickle=False)
    np.save(os.path.join(args.save_dir, 'target.npy'), target_array[:40000], allow_pickle=False)
    np.save(os.path.join(args.save_dir, 'data.npy'), data_array, allow_pickle=False)
generate_feature()
