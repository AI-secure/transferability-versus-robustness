import torch
import torch.nn as nn
import torch.nn.functional as F

def matrix_power3(inp):
    return torch.bmm(torch.bmm(inp, inp), inp)

def OLM_normalize(weight, eps=1e-4): # TODO: eps=1e-5
    Z = weight.view(weight.shape[0], -1) # 
    Zc = Z - Z.mean(dim=-1, keepdim=True)
    S = torch.matmul(Zc, Zc.transpose(0, 1))
    eye = torch.eye(S.shape[-1]).to(S)
    S = S + eps*eye
    #norm_S = S.norm(p=None, dim=(0,1))
    #S = S.div(norm_S)
    SIGMA_diag, D = torch.symeig(S, eigenvectors=True)
    SIGMA_msqrt = torch.diag(1.0 / torch.sqrt(SIGMA_diag))
    W = D @ SIGMA_msqrt @ D.T @ Zc
    #W = SIGMA_msqrt @ D.T
    return W.view_as(weight)

def ONI_normalize(weight, T=5, norm_groups=1, eps=1e-5):
    assert weight.shape[0] % norm_groups == 0
    Z = weight.view(norm_groups, weight.shape[0]//norm_groups, -1) # 
    Zc = Z - Z.mean(dim=-1, keepdim=True)
    S = torch.matmul(Zc, Zc.transpose(1, 2))
    eye = torch.eye(S.shape[-1]).to(S).expand(S.shape)
    S = S + eps*eye
    #norm_S = S.norm(p='fro', dim=(1, 2), keepdim=True)
    norm_S = S.norm(p=None, dim=(1, 2), keepdim=True)
    S = S.div(norm_S)
    B = [torch.Tensor([]) for _ in range(T + 1)]
    B[0] = torch.eye(S.shape[-1]).to(S).expand(S.shape)
    for t in range(T):
        #B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, torch.matrix_power(B[t], 3), S)
        B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, matrix_power3(B[t]), S)
    W = B[T].matmul(Zc).div_(norm_S.sqrt())
    #print(W.matmul(W.transpose(1,2)))
    # W = oni_py.apply(weight, self.T, ctx.groups)
    return W.view_as(weight)

class ONI_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super(ONI_Linear, self).__init__(in_features, out_features, bias)
        #assert in_features >= out_features
        self.scale = scale

    def forward(self, input_f:torch.Tensor) -> torch.Tensor:
        weight_q = self.normed_weight()
        return F.linear(input_f, weight_q, self.bias)

    def normed_weight(self):
        weight_q = ONI_normalize(self.weight)
        weight_q = weight_q * self.scale
        return weight_q

class ONI_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                scale=1.0):
        super(ONI_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        assert in_channels*kernel_size*kernel_size >= out_channels
        self.scale = scale

    def forward(self, input_f:torch.Tensor) -> torch.Tensor:
        weight_q = self.normed_weight()
        return F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def normed_weight(self):
        #print (self.weight.shape)
        #print (torch.mm(self.weight.view(self.weight.shape[0],-1), self.weight.view(self.weight.shape[0],-1).T))
        weight_q = ONI_normalize(self.weight)
        #print (weight_q.shape)
        #print (torch.mm(weight_q.view(weight_q.shape[0],-1), weight_q.view(weight_q.shape[0],-1).T))
        weight_q = weight_q * self.scale
        #print (weight_q.shape)
        #print (torch.mm(weight_q.view(weight_q.shape[0],-1), weight_q.view(weight_q.shape[0],-1).T))
        #assert 0
        return weight_q

class INFL_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super(INFL_Linear, self).__init__(in_features, out_features, bias)
        assert in_features >= out_features
        self.scale = scale

    def forward(self, input_f:torch.Tensor) -> torch.Tensor:
        weight_q = self.normed_weight()
        return F.linear(input_f, weight_q, self.bias)

    def normed_weight(self):
        rowsum = self.weight.abs().sum(1,keepdims=True)
        max_rowsum = torch.FloatTensor([self.scale])
        if self.weight.is_cuda:
            max_rowsum = max_rowsum.cuda()
        rowsum_adjust = torch.min(rowsum, max_rowsum)
        weight_q = self.weight * (rowsum_adjust/rowsum)
        #weight_q = self.weight / (rowsum+1e-8) * self.scale
        return weight_q

class INFL_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                scale=1.0):
        super(INFL_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        assert in_channels*kernel_size*kernel_size >= out_channels
        self.scale = scale

    def forward(self, input_f:torch.Tensor) -> torch.Tensor:
        weight_q = self.normed_weight()
        return F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def normed_weight(self):
        Cout, Cin, H, W = self.weight.shape
        assert H==W
        w_flat = self.weight.view(Cout, Cin*H*W)
        rowsum = w_flat.abs().sum(1,keepdims=True)
        max_rowsum = torch.FloatTensor([self.scale])
        if self.weight.is_cuda:
            max_rowsum = max_rowsum.cuda()
        rowsum_adjust = torch.min(rowsum, max_rowsum)
        weight_q = w_flat * (rowsum_adjust/rowsum)
        weight_q = weight_q.view(Cout,Cin,H,W)
        return weight_q

def spectral_normalize(weight, u, v, T=5):
    Z = weight.view(weight.shape[0], -1)
    for t in range(T):
        u = torch.mm(Z,v)
        v = torch.mm(Z.T,u)
        u = u / u.norm()
        v = v / v.norm()
    W = Z / torch.mm(u.T,torch.mm(Z,v))
    return W.view_as(weight), u, v
    
class SPECTRAL_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super(SPECTRAL_Linear, self).__init__(in_features, out_features, bias)
        assert in_features >= out_features
        self.scale = scale

        self.spec_u = nn.Parameter(torch.FloatTensor(out_features,1).normal_())
        self.spec_u.requires_grad = False
        self.spec_v = nn.Parameter(torch.FloatTensor(in_features,1).normal_())
        self.spec_v.requires_grad = False

    def forward(self, input_f:torch.Tensor) -> torch.Tensor:
        weight_q = self.normed_weight()
        return F.linear(input_f, weight_q, self.bias)

    def normed_weight(self):
        weight_q, new_u, new_v = spectral_normalize(self.weight, self.spec_u, self.spec_v)
        if self.training:
            self.spec_u.data = new_u
            self.spec_v.data = new_v
        weight_q = weight_q * self.scale
        return weight_q
