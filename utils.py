import torch
import torchvision
import torchvision.transforms as transforms
from advertorch.utils import NormalizeByChannelMeanStd

def load_data(train_batch_size=128, test_batch_size=100, train_aug=True, complex_aug=False):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).cuda()
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).cuda()
    normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)

    if train_aug:
        if complex_aug:
            color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            transform_train = transforms.Compose([
                rnd_color_jitter,
                rnd_gray,
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(32),
                transforms.ToTensor(),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./raw_data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./raw_data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    #classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #           'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset, testset, trainloader, testloader, normalizer


def load_stl_data(train_batch_size=128, test_batch_size=100):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).cuda()
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).cuda()
    normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)

    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),#
        #transforms.RandomHorizontalFlip(),#
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.STL10(
        root='./raw_data', split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.STL10(
        root='./raw_data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    return trainset, testset, trainloader, testloader, normalizer

def load_svhn_data(train_batch_size=128, test_batch_size=100):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).cuda()
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).cuda()
    normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)

    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),#
        #transforms.RandomHorizontalFlip(),#
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.SVHN(
        root='./raw_data', split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.SVHN(
        root='./raw_data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    return trainset, testset, trainloader, testloader, normalizer

def load_pets_data(train_batch_size=128, test_batch_size=100):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).cuda()
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).cuda()
    normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)

    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),#
        #transforms.RandomHorizontalFlip(),#
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder('./raw_data/pets/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.ImageFolder('./raw_data/pets/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    return trainset, testset, trainloader, testloader, normalizer

def load_flowers_data(train_batch_size=128, test_batch_size=100):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).cuda()
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).cuda()
    normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)

    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),#
        #transforms.RandomHorizontalFlip(),#
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder('./raw_data/flowers_new/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.ImageFolder('./raw_data/flowers_new/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    return trainset, testset, trainloader, testloader, normalizer


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def calc_repr(model, x):
    assert isinstance(model, torchvision.models.ResNet)
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x
