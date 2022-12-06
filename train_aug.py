import torch
import torch.nn as nn
from tqdm import tqdm

from utils import load_data
from models import ResNet18, CIFAR_CNN

model_arch = 'cifarcnn'
#model_arch = 'resnet18'

#complex_aug=0
complex_aug=3
aug_param=45
#complex_aug=4
#aug_param=0.2
#complex_aug=5
#aug_param=30
#complex_aug=8
#aug_param=4
#complex_aug=9
#aug_param=0.25
#complex_aug=12
#aug_param=0.25
SAVE_PATH = './saved_model/aug%s.pth'

if complex_aug == 3:
    SAVE_PATH = SAVE_PATH%('_rotate%s'%aug_param)
elif complex_aug == 4:
    SAVE_PATH = SAVE_PATH%('_erase%s'%aug_param)
elif complex_aug == 5:
    SAVE_PATH = SAVE_PATH%('_affine%s'%aug_param)
elif complex_aug == 8:
    SAVE_PATH = SAVE_PATH%('_posterize%d'%aug_param)
elif complex_aug == 9:
    SAVE_PATH = SAVE_PATH%('_perspective%s'%aug_param)
elif complex_aug == 12:
    SAVE_PATH = SAVE_PATH%('_translate%s'%aug_param)
else:
    assert complex_aug == 0
    SAVE_PATH = SAVE_PATH%'_standard'

if model_arch != 'resnet18':
    SAVE_PATH = SAVE_PATH[:-4] + '-%s'%model_arch + '.pth'
print (SAVE_PATH)


from advertorch.utils import NormalizeByChannelMeanStd
import torchvision
import torchvision.transforms as transforms
mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).cuda()
std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).cuda()
normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)

if complex_aug == 3:
    transform_train = transforms.Compose([
        transforms.RandomRotation(aug_param),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
elif complex_aug == 4:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(scale=(0.02,aug_param)),
    ])
elif complex_aug == 5:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomAffine(degrees=aug_param,translate=(0.1,0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
elif complex_aug == 8:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomPosterize(bits=aug_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
elif complex_aug == 9:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomPerspective(distortion_scale=aug_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
elif complex_aug == 12:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomAffine(degrees=0,translate=(aug_param,aug_param)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
else:
    assert complex_aug == 0
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='./raw_data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./raw_data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
print (len(trainset), len(testset))



if model_arch == 'resnet18':
    model = ResNet18(normalizer, dropout=0.0)
elif model_arch == 'cifarcnn':
    model = CIFAR_CNN(normalizer, dropout=0.0)
model = model.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=(0.01 if model_arch == 'cifarcnn' else 0.1), momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(trainloader) as pbar:
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to('cuda'), y.to('cuda')
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred_c = pred.max(1)
            total += y.size(0)
            correct += pred_c.eq(y).sum().item()
            pbar.set_description('Loss: %.3f | Acc:%.3f%%'%(train_loss/(batch_idx+1), 100.*correct/total))

    acc = 100.*correct/total
    return train_loss/len(trainloader), acc

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(), tqdm(testloader) as pbar:
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to('cuda'), y.to('cuda')
            pred = model(x)
            loss = criterion(pred, y)

            test_loss += loss.item()
            _, pred_c = pred.max(1)
            total += y.size(0)
            correct += pred_c.eq(y).sum().item()
            pbar.set_description('Loss: %.3f | Acc:%.3f%%'%(test_loss/(batch_idx+1), 100.*correct/total))

    acc = 100.*correct/total
    return test_loss/len(testloader), acc


best_acc = 0.0
for epoch in range(200):
    train(epoch)
    _, cur_acc = test(epoch)
    scheduler.step()
    if cur_acc > best_acc:
        best_acc = cur_acc
        torch.save(model.state_dict(), SAVE_PATH)
