import numpy as np
import torch
import torch.nn.functional as F

class LinfPGDAttack(object):
    def __init__(self, model, num_steps=10, epsilon=8./255.):
        self.model = model
        self.num_steps = num_steps
        self.epsilon = epsilon
        self.alpha = epsilon/4

    def perturb(self, x, y):
        origin_training = self.model.training
        self.model.eval()

        adv_x = x.detach()
        adv_x = adv_x + torch.zeros_like(adv_x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            adv_x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(adv_x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [adv_x])[0]
            adv_x = adv_x.detach() + self.alpha * torch.sign(grad.detach())
            adv_x = torch.min(torch.max(adv_x, x-self.epsilon), x+self.epsilon)
            adv_x = torch.clamp(adv_x,0,1)
        if origin_training:
            self.model.train()
        return adv_x.detach()

class L2PGDAttack(object):
    def __init__(self, model, num_steps=10, epsilon=1.0, alpha=None):
        self.model = model
        self.num_steps = num_steps
        self.epsilon = epsilon
        #self.alpha = alpha
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = epsilon/4

    def perturb(self, x, y, normalize=None, resample=False):
        #origin_training = self.model.training
        #self.model.eval()
        B = x.shape[0]

        if normalize is not None:
            mean, std = torch.cuda.FloatTensor(normalize[0]).view(1,3,1,1), torch.cuda.FloatTensor(normalize[1]).view(1,3,1,1)
            #adv_x = (x*std+mean).detach()
            x = (x*std+mean).detach()
        else:
            mean, std = 0, 1
            #adv_x = x.detach()

        adv_x = x.detach()

        adv_x = adv_x + torch.zeros_like(adv_x).uniform_(-self.epsilon, self.epsilon)
        delta = adv_x - x
        delta_norm = delta.view(B, -1).norm(dim=1).view(-1,1,1,1)
        clamp_delta = delta / delta_norm * torch.clamp(delta_norm,0,self.epsilon)
        adv_x = x + clamp_delta
        adv_x = torch.clamp(adv_x,0,1)

        for i in range(self.num_steps):
            if resample:
                self.model.unfix_noise()
                self.model.fix_noise(N_smooth=64)
            adv_x.requires_grad_()
            with torch.enable_grad():
                logits = self.model((adv_x-mean)/std)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [adv_x])[0]
            grad_norm = grad.view(B, -1).norm(dim=1).view(-1,1,1,1)
            grad = grad / grad_norm
            adv_x = adv_x.detach() + self.alpha * grad.detach()

            delta = adv_x - x
            delta_norm = delta.view(B, -1).norm(dim=1).view(-1,1,1,1)
            clamp_delta = delta / delta_norm * torch.clamp(delta_norm,0,self.epsilon)
            adv_x = x + clamp_delta
            adv_x = torch.clamp(adv_x,0,1)
        #if origin_training:
        #    self.model.train()
        return ((adv_x-mean)/std).detach()
