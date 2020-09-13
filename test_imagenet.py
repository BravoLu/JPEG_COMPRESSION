'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.datasets import ImageFolder
from dataset import *
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = ImageNette(transform=transform_test)
#testset = ImageNette(transform=transform_test, root='/raid/home/bravolu/data/resize_imagenette2/val')
#testset = ImageNette(transform=transform_test, root='/raid/home/bravolu/data/imagenette2_png/val')
print(torchvision.__file__)
print(len(testset))
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=False, num_workers=2)

#classes = ('plane', 'car', 'bird', 'cat', 'deer',
#           'dog', 'frog', 'horse', 'ship', 'truck')

classes = ['Tench', 'English Springer', 'Cassette Player', 'Chain Saw', 'Church',
           'French Horn', 'Garbage Truck', 'Gas Pump', 'Golf Ball', 'Parachute']

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = EfficientNetB0()
# net = RegNetX_200MF()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
#net.load_state_dict(torch.load("checkpoint/imagenette_ResNet_lack.pth")['net'])
#net.load_state_dict(torch.load("checkpoint/imagenette_ResNet_resize_lack_v2.pth")['net'])
#net.load_state_dict(torch.load("checkpoint/imagenette_ResNet_v2.pth")['net'])
#net.load_state_dict(torch.load("checkpoint/imagenette_ResNet_png.pth")['net'])
#net.load_state_dict(torch.load("checkpoint/imagenette_EfficientNet_resize_jpeg.pth")['net'])
net.load_state_dict(torch.load("checkpoint/imagenette_EfficientNet.pth")['net'])

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    corrects = [0 for _ in range(10)]
    amounts =[0 for _ in range(10)]

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            #correct += predicted.eq(targets).sum().item()
            for label in range(10):
                #import pdb
                #pdb.set_trace()
                corrects[label] += predicted[targets.eq(label)].eq(label).sum().item()
                amounts[label] += targets.eq(label).sum().item()


    # Save checkpoint.
    acc = 100.*sum(corrects)/sum(amounts)
    print('acc {:.2f}%'.format(acc))
    for i in range(10):
        print("{} {:.2f}%".format(classes[i], 100*corrects[i]/amounts[i]))

    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/%s_%s_latest.pth'%('cifar10', net.module.__class__.__name__))
    #     best_acc = acc


test(0)
# for epoch in range(start_epoch, start_epoch+350):
#     train(epoch)
#     test(epoch)
