import torch
import torch.nn as nn
from tqdm import tqdm

from utils import load_data
from models import ResNet18, CIFAR_CNN

model_arch = 'cifarcnn'
#model_arch = 'resnet18'

if model_arch != 'resnet18':
    SAVE_PATH = './saved_model/vanilla-%s.pth'%model_arch
else:
    SAVE_PATH = './saved_model/vanilla.pth'

trainset, testset, trainloader, testloader, normalizer = load_data()
print (len(trainset), len(testset))

if model_arch == 'resnet18':
    model = ResNet18(normalizer, dropout=0.0)
elif model_arch == 'cifarcnn':
    model = CIFAR_CNN(normalizer, dropout=0.0)
model = model.to('cuda')
print (model)

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
