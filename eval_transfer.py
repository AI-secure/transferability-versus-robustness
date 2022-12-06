import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from utils import load_stl_data, load_svhn_data, load_pets_data, load_flowers_data
from attack_lib import LinfPGDAttack, L2PGDAttack
from models import ResNet18, ResNet50, ResNet152, CIFAR_CNN

def main():
    MODEL_NAME = 'vanilla-cifarcnn'

    trainset, testset, trainloader, testloader, normalizer = load_svhn_data()
    out_dim = 10
    print (MODEL_NAME, len(trainset), len(testset))
    #assert 0

    if 'cifarcnn' in MODEL_NAME:
        model = CIFAR_CNN(normalizer, dropout=0.0)
    else:
        model = ResNet18(normalizer, dropout=0.0)
    model = model.to('cuda')
    model.load_state_dict(torch.load('./saved_model/%s.pth'%MODEL_NAME))
    print (model.linear)
    model.linear = nn.Linear(model.linear.in_features, out_dim).to('cuda')
    model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=(0.01 if 'cifarcnn' in MODEL_NAME else 0.1), momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)

    best_acc = 0.0
    for epoch in range(40):
        # Train
        print ("Epoch %d"%epoch)
        model.eval()
        #model.train()
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
        scheduler.step()

        # Test
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
        cur_acc = 100.*correct/total

        # Save
        if cur_acc > best_acc:
            best_acc = cur_acc
            torch.save(model.state_dict(), './saved_model/%s-transfer.pth'%(MODEL_NAME))


if __name__ == '__main__':
    main()
