import argparse

import torch.optim as optim
from torch.ao.quantization import QConfig, FusedMovingAvgObsFakeQuantize, MovingAverageMinMaxObserver, prepare_qat, \
    MovingAveragePerChannelMinMaxObserver
import model_filter_basis

from training_functions import *
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.ao.nn.intrinsic as nni

import os
import random
from torchsummary import summary


def set_seed(seed):
    # Set seed for python libraries
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set seed for pytorch
    torch.manual_seed(seed)  # Set seed for CPU
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    torch.cuda.manual_seed(seed)  # Set seed for the current GPU
    torch.cuda.manual_seed_all(seed)  # Set seed for all the GPUs
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

##########################
### Cifar100 Dataset
##########################
def create_datasets(batch_size):
    # dataset setting
    num_workers = 2
    Cifar100_root = '~/Cifar100'
    CIFAR100_TRAIN_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_TRAIN_STD = (0.2675, 0.2565, 0.2761)

    # CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
    # CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)
    # dataset setting for Imagenet
    transform_training = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])


    # define training and test datasets
    trainset = datasets.CIFAR100(root=Cifar100_root, train=True,
                                download=True, transform=transform_training)

    valset = datasets.CIFAR100(root=Cifar100_root, train=False,
                              download=True, transform=transform_val)
    train_loader = DataLoader(trainset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_float', default=0.05, type=float,
                        help='learning rate for float training (default: 0.05)')
    parser.add_argument('--epochs_float', default=300, type=int,
                        help='epochs for float training(default: 300)')
    parser.add_argument('--lr_qat', default=0.0005, type=float,
                        help='learning rate for QAT (default: 0.0005)')
    parser.add_argument('--epochs_qat', default=50, type=int,
                        help='epochs for QAT(default: 50)')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size (default: 64)')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device (default: cuda)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed (default: 0)')
    parser.add_argument('--path_save', default='checkpoint_Cifar100_vgg16', type=str,
                        help='save dir (default: save)')
    parser.add_argument('--filter_basis_channel', default=1, type=int,
                        help='channel for filter basis (default: 1)')
    parser.add_argument('--filter_basis_number', default=5, type=int,
                        help='number for filter basis (default: 5)')
    parser.add_argument('--filter_basis_weights_quant_bits', default=8, type=int,
                        help='bits for filter basis weights (default: 8)')
    parser.add_argument('--coefficients_quant_bits', default=8, type=int,
                        help='bits for coefficients of linear combination (default: 8)')
    parser.add_argument('--activations_quant_bits', default=8, type=int,
                        help='bits for activations (default: 8)')
    parser.add_argument('--coefficients_shift_en', default=0, type=int,
                        help='enable bitwise shift operation for coefficients (default: 0)')
    args = parser.parse_args()

    device=args.device
    learning_rate_float = args.lr_float
    learning_rate_qat=args.lr_qat
    path= args.path_save
    epochs_float = args.epochs_float
    epochs_qat = args.epochs_qat
    seed = args.seed
    if args.coefficients_shift_en==0:
        shift_en = False
    else:
        shift_en = True
    set_seed(seed)

    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = create_datasets(args.batch_size)
    #creat compressed model
    model1 = model_filter_basis.VGG16_shared_basis_filter(100,32,args.filter_basis_channel,args.filter_basis_number,shift=shift_en)
    model1.to("cpu")
    summary(model1, (3, 32, 32),device="cpu")
    print("device:" + device)
    print("seed:" + str(seed))
    print("lr:" + str(learning_rate_float))
    print("epochs:" + str(epochs_float))
    print("shift:" + str(shift_en))
    print("path:" + path)
    #Train float model
    print("Train float model")
    optimizer = optim.SGD(model1.parameters(), lr=learning_rate_float, momentum=0.9, weight_decay=5e-4)
    model1.to(device)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_float)
    train(model1, device, train_loader, val_loader, optimizer, epochs_float, criterion, epochs_float, 0.1,
          path=path + '.pt', train_scheduler=train_scheduler)
    #Train quantized model
    print("Train quantized model")
    print("lr:" + str(learning_rate_qat))
    print("epochs:" + str(epochs_qat))
    # set quantization config for linear layers
    model1.qconfig = QConfig(
        activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                           quant_min=0,
                                                           quant_max=255,
                                                           dtype=torch.quint8,
                                                           qscheme=torch.per_tensor_affine),
        weight=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                       quant_min=-127,
                                                       quant_max=127,
                                                       dtype=torch.qint8,
                                                       qscheme=torch.per_tensor_symmetric))
    # fuse BatchNorm with ReLU , linear with Relu
    patterns = model_filter_basis.find_specific_patterns(model1)
    model1 = torch.ao.quantization.fuse_modules_qat(model1, patterns)
    #set quantization config for filter basis and coefficients
    modules = list(model1.named_modules())
    for i in range(len(modules)):
        name1, module1 = modules[i]
        # set quantization config for activations of coefficients
        if isinstance(module1, nni.BNReLU2d):
            module1.qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                              quant_min=0,
                                                                              quant_max=2**args.activations_quant_bits-1,
                                                                              dtype=torch.quint8,
                                                                              qscheme=torch.per_tensor_affine),weight=None)
        # set quantization config for weights of coefficients
        if isinstance(module1, nn.Conv2d):
            module1.qconfig = QConfig(activation=torch.nn.Identity,
                                    weight=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                                                      quant_min=-(2**(args.coefficients_quant_bits-1)),
                                                                                      quant_max=2**(args.coefficients_quant_bits-1)-1,
                                                                                      dtype=torch.qint8,
                                                                                      qscheme=torch.per_channel_symmetric))
        # set quantization config for weights and activations of filter basis
        if isinstance(module1, model_filter_basis.conv_shared_filter_basis):
            module1.qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                  quant_min=0,
                                                                                  quant_max=2**args.activations_quant_bits-1,
                                                                                  dtype=torch.quint8,
                                                                                  qscheme=torch.per_tensor_affine),
                                weight=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                  quant_min=-(2**(args.filter_basis_weights_quant_bits-1)),
                                                                                  quant_max=2**(args.filter_basis_weights_quant_bits-1)-1,
                                                                                  dtype=torch.qint8,
                                                                                  qscheme=torch.per_tensor_symmetric))
    model1.train()
    # convert floating model to fake quantized
    prepare_qat(model1, inplace=True, mapping=model_filter_basis.prepare_custom_config_dict)

    optimizer = optim.SGD(model1.parameters(), lr=learning_rate_qat, momentum=0.9, weight_decay=5e-4)
    model1.to(device)
    train_scheduler = optim.lr_scheduler.StepLR(optimizer, 5,0.1)
    train(model1, device, train_loader, val_loader, optimizer, epochs_qat, criterion, epochs_qat, 0.01,
          path=path + '_quantized.pt', train_scheduler=train_scheduler)


