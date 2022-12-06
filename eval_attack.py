import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from utils import load_data
from attack_lib import LinfPGDAttack, L2PGDAttack
from models import ResNet18, ResNet50, ResNet152, CIFAR_CNN

def main():
    MODEL_NAME = 'vanilla-cifarcnn'

    test_batch_size=100
    trainset, testset, trainloader, testloader, normalizer = load_data(test_batch_size=test_batch_size)
    print (MODEL_NAME, len(trainset), len(testset))

    if 'cifarcnn' in MODEL_NAME:
        model = CIFAR_CNN(normalizer, dropout=0.0)
    else:
        model = ResNet18(normalizer, dropout=0.0)
    model = model.to('cuda')
    model.load_state_dict(torch.load('./saved_model/%s.pth'%MODEL_NAME))
    model.eval()

    adversary = L2PGDAttack(model, epsilon=0.25, num_steps=20)

    correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad(), tqdm(testloader) as pbar:
        for x, y in pbar:
            x, y = x.to('cuda'), y.to('cuda')
            adv_x = adversary.perturb(x, y).detach()

            adv_pred = model(adv_x)
            _, adv_pred_c = adv_pred.max(1)
            total += y.size(0)
            adv_correct += adv_pred_c.eq(y).sum().item()
            pbar.set_description('Adv acc: %.3f'%(100.*adv_correct/total))


if __name__ == '__main__':
    main()
