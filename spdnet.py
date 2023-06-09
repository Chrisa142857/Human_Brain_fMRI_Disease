# import torch
# from torch import nn
# from torch.optim.optimizer import Optimizer
# from torch.autograd import Function
# import numpy as np
# import time

# class StiefelParameter(nn.Parameter):
#     """A kind of Variable that is to be considered a module parameter on the space of 
#         Stiefel manifold.
#     """
#     def __new__(cls, data=None, requires_grad=True):
#         return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

#     def __repr__(self):
#         return 'Parameter containing:' + self.data.__repr__()
    
# """
# Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
# """

# # BiMap layer
# class SPDTransform(nn.Module):

#     def __init__(self, input_size, output_size, in_channels=1):
#         super(SPDTransform, self).__init__()

#         # temp = torch.load('temp/weight_%d_%d.pt' % (input_size, output_size))
#         if in_channels > 1:
#             self.weight = StiefelParameter(torch.FloatTensor(in_channels, input_size, output_size), requires_grad=True)
#         else:
#             self.weight = StiefelParameter(torch.FloatTensor(input_size, output_size), requires_grad=True)
#         # self.weight = StiefelParameter(temp, requires_grad=True)
#         # print(self.weight)
#         nn.init.orthogonal_(self.weight)  # W需满足半正交，位于施蒂费尔流形上
#         # torch.save(self.weight, 'temp/weight_%d_%d.pt' % (input_size, output_size))

#     def forward(self, input):
#         # 将输入矩阵升维

#         weight = self.weight

#         # for i in input.shape[:-2]:
#         #     weight = weight.unsqueeze(0)
#         # weight = weight.expand(*(input.shape[:-2]), -1, -1)

#         output = weight.transpose(-2, -1) @ input @ weight
#         return output

# """
# Yu, K., & Salzmann, M. (2017). Second-order convolutional neural networks. arXiv preprint arXiv:1703.06817.
# """

# class ParametricVectorize(nn.Module):

#     def __init__(self, input_size, output_size):
#         super(ParametricVectorize, self).__init__()
#         self.weight = nn.Parameter(torch.ones(output_size, input_size), requires_grad=True)

#     def forward(self, input):
#         weight = self.weight.unsqueeze(0)
#         weight = weight.expand(input.size(0), -1, -1)
#         output = torch.bmm(weight, input)
#         output = torch.bmm(output, weight.transpose(1, 2))
#         output = torch.mean(output, 2)
#         return output

# """
# Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
# """

# # 向量化
# class SPDVectorize(nn.Module):

#     def __init__(self, input_size, vectorize_all=True):
#         super(SPDVectorize, self).__init__()
#         row_idx, col_idx = np.triu_indices(input_size)
#         self.register_buffer('row_idx', torch.LongTensor(row_idx))
#         self.register_buffer('col_idx', torch.LongTensor(col_idx))
#         self.register_buffer('vectorize_all', torch.tensor(vectorize_all))

#     def forward(self, input):
#         output = input[..., self.row_idx, self.col_idx]

#         if self.vectorize_all:
#             output = torch.flatten(output, 1)
#         return output

# """
# Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
# """

# class SPDTangentSpaceFunction(Function):

#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)

#         output = input.new(input.size(0), input.size(1), input.size(2))
#         # 特征值取log
#         for k, x in enumerate(input):
#             # u, s, v = x.svd()
#             s, u = x.symeig(eigenvectors=True)
#             s.log_()
#             output[k] = u.mm(s.diag().mm(u.t()))

#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input = ctx.saved_variables
#         input = input[0]
#         grad_input = None

#         if ctx.needs_input_grad[0]:
#             eye = input.new(input.size(1))
#             eye.fill_(1)
#             eye = eye.diag()
#             grad_input = input.new(input.size(0), input.size(1), input.size(1))
#             for k, g in enumerate(grad_output):
#                 x = input[k]
#                 # u, s, v = x.svd()
#                 u, s, _ = torch.linalg.svd(x, full_matrices=True)
#                 # s, u = x.symeig(eigenvectors=True)

#                 g = symmetric(g)

#                 s_log_diag = s.log().diag()
#                 s_inv_diag = (1 / s).diag()

#                 dLdV = 2 * (g.mm(u.mm(s_log_diag)))
#                 dLdS = eye * (s_inv_diag.mm(u.t().mm(g.mm(u))))

#                 P = s.unsqueeze(1)
#                 P = P.expand(-1, P.size(0))
#                 P = P - P.t()

#                 # mask_zero = torch.abs(P) == 0
#                 # P = 1 / P
#                 # P[mask_zero] = 0

#                 index_diag = np.diag_indices(P.shape[0])
#                 P = 1 / P
#                 P[index_diag[0], index_diag[1]] = 0
#                 P[P.isinf()] = 0

#                 grad_input[k] = u.mm(symmetric(P.t() * (u.t().mm(dLdV))) + dLdS).mm(u.t())
#                 # grad_input[k] = u.mm(P.t() * symmetric(u.t().mm(dLdV)) + dLdS).mm(u.t())

#         return grad_input

# """
# Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
# """
# grads = []

# def save_grad(name):
#     def hook(grad):
#         grads.append(grad)

#     return hook

# class SPDTangentSpace(nn.Module):

#     def __init__(self, input_size, vectorize=True, vectorize_all=True):
#         super(SPDTangentSpace, self).__init__()
#         self.vectorize = vectorize
#         if vectorize:
#             self.vec = SPDVectorize(input_size, vectorize_all=vectorize_all)

#     def forward(self, input):
#         # output = SPDTangentSpaceFunction.apply(input)

#         # u, s, v = input.svd()
#         # s = s.log().diag_embed()
#         # output = u @ s @ u.transpose(-2, -1)

#         s, u = torch.linalg.eigh(input)
#         s = s.log().diag_embed()
#         output = u @ s @ u.transpose(-2, -1)

#         # input.register_hook(save_grad('input'))

#         if self.vectorize:
#             output = self.vec(output)

#         return output

# """
# Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
# """

# class SPDRectifiedFunction(Function):

#     @staticmethod
#     def forward(ctx, input, epsilon):
#         ctx.save_for_backward(input, epsilon)
#         # 特征值ReLU
#         # u, s, v = input.svd()
#         u, s, _ = torch.linalg.svd(input, full_matrices=False)
#         s[s < epsilon[0]] = epsilon[0]
#         s = torch.diag_embed(s)
#         output = s @ u.transpose(-2, -1) @ u
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, epsilon = ctx.saved_variables
#         grad_input = None

#         if ctx.needs_input_grad[0]:
#             eye = input.new(input.size(-1))
#             eye.fill_(1)
#             eye = eye.diag()

#             g = grad_output
#             g = symmetric(g)

#             # u, s, v = input.svd()
#             u, s, _ = torch.linalg.svd(input, full_matrices=True)

#             max_mask = s > epsilon
#             s_max_diag = s.clone()
#             s_max_diag[~max_mask] = epsilon
#             s_max_diag = s_max_diag.diag_embed()
#             Q = max_mask.float().diag_embed()

#             dLdV = 2 * (g @ u @ s_max_diag)
#             dLdS = eye * (Q @ u.transpose(-2, -1) @ g @ u)

#             P = s.unsqueeze(-2)
#             P = P - P.transpose(-2, -1)

#             index_zero = np.diag_indices(P.shape[-1])
#             P = 1 / P
#             P[..., index_zero[0], index_zero[1]] = 0
#             P[P.isinf()] = 0  # 防止对角线以外的元素为inf

#             grad_input = u @ symmetric(P.transpose(-2, -1) * (u.transpose(-2, -1) @ dLdV) + dLdS) @ u.transpose(-2, -1)

#         return grad_input, None
#     # return grad_input

# """
# Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
# """

# class SPDRectified(nn.Module):

#     def __init__(self, epsilon=1e-4):
#         super(SPDRectified, self).__init__()
#         self.register_buffer('epsilon', torch.FloatTensor([epsilon]))

#     def forward(self, input):
#         output = SPDRectifiedFunction.apply(input, self.epsilon)
#         return output

# class SPDNormalization(nn.Module):
#     def __init__(self):
#         super(SPDNormalization, self).__init__()

#     def forward(self, input):
#         fro_norm = torch.norm(input, p='fro', dim=[-2, -1])
#         fro_norm = fro_norm.unsqueeze(-1).unsqueeze(-1).expand_as(input)
#         output = input / fro_norm
#         return output

# class SPDConv(nn.Module):

#     def __init__(self, in_channels, out_channels, kernel_size, epsilon=1e-4):
#         super(SPDConv, self).__init__()
#         self.register_buffer('epsilon', torch.FloatTensor([epsilon]))
#         if isinstance(kernel_size, int):
#             kernel_size = (kernel_size, kernel_size)
#         self.v = nn.Parameter(torch.randn((out_channels, in_channels) + kernel_size), requires_grad=True)
#         nn.init.xavier_uniform_(self.v)
#         # self.weight = nn.Parameter(torch.randn((out_channels, in_channels) + kernel_size), requires_grad=True)

#     def forward(self, input):
#         # if len(input.shape) < 4:
#         #     input = input.unsqueeze(1)
#         weight = torch.matmul(self.v.transpose(-2, -1), self.v) + self.epsilon[0] * torch.eye(self.v.shape[-1],
#                                                                                               device=input.device)
#         return nn.functional.conv2d(input.float(), weight=weight)

# class SPDActivate(nn.Module):
#     def __init__(self, activate_func='sinh'):
#         super(SPDActivate, self).__init__()
#         if activate_func == 'sinh':
#             self.activate_func = torch.sinh
#         elif activate_func == 'cosh':
#             self.activate_func = torch.cosh
#         else:
#             self.activate_func = torch.exp

#     def forward(self, input):
#         output = self.activate_func(input)
#         return output

# class SPDDiag(nn.Module):
#     def __init__(self, activate_func='sinh'):
#         super(SPDDiag, self).__init__()
#         if activate_func == 'sinh':
#             self.activate_func = torch.sinh
#         elif activate_func == 'cosh':
#             self.activate_func = torch.cosh
#         else:
#             self.activate_func = torch.exp

#     def forward(self, input):
#         output = self.activate_func(input)
#         output = torch.flatten(output, 1)
#         output = output.log()
#         return output

# class Normalize(nn.Module):
#     def __init__(self, p=2, dim=-1):
#         super(Normalize, self).__init__()
#         self.p = p
#         self.dim = dim

#     def forward(self, input):
#         output = torch.nn.functional.normalize(input, p=self.p, dim=self.dim)
#         return output


# def symmetric(A):
#     return 0.5 * (A + A.transpose(-2, -1))

# class StiefelMetaOptimizer(object):
#     """This is a meta optimizer which uses other optimizers for updating parameters
#         and remap all StiefelParameter parameters to Stiefel space after they have been updated.
#     """

#     def __init__(self, optimizer):
#         self.optimizer = optimizer
#         self.state = {}


#     def zero_grad(self):
#         return self.optimizer.zero_grad()

#     def state_dict(self):
#         return self.optimizer.state_dict()

#     @torch.no_grad()
#     def step(self, closure=None):
#         """Performs a single optimization step.

#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """

#         for group in self.optimizer.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 if isinstance(p, StiefelParameter):
#                     # state存储p的原始数据
#                     # if id(p) not in self.state:
#                     #     self.state[id(p)] = p.data.clone()
#                     # else:
#                     #     self.state[id(p)].fill_(0).add_(p.data)

#                     # p.data.fill_(0)

#                     # 求p的黎曼梯度
#                     trans = orthogonal_projection(p.grad, p)
#                     p.grad.fill_(0).add_(trans)

#         # 根据梯度更新参数，包括黎曼梯度和欧式梯度
#         loss = self.optimizer.step(closure)

#         for group in self.optimizer.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 if isinstance(p, StiefelParameter):
#                     # 更新后的p再映射回施蒂费尔流形
#                     # trans = retraction(p.data, self.state[id(p)])
#                     trans = retraction(p)
#                     p.fill_(0).add_(trans)

#         return loss

# # 求B的黎曼梯度，A为欧式梯度
# def orthogonal_projection(A, B):
#     out = A - B.mm(symmetric(B.transpose(0, 1).mm(A)))
#     return out

# # 将切向量从切空间映射回施蒂费尔流形
# def retraction(A, ref=None):
#     # ref为None时，A已经是原切点与切向量之和
#     if ref == None:
#         data = A
#     else:
#         data = A + ref
#     # Q, R = data.qr()
#     Q, R = torch.linalg.qr(data)
#     # To avoid (any possible) negative values in the output matrix, we multiply the negative values by -1
#     sign = (R.diag().sign() + 0.5).sign().diag()
#     out = Q.mm(sign)
#     return out


'''
    Original SPDNet code:
'''

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.autograd import Function
import numpy as np


class StiefelParameter(nn.Parameter):
    """A kind of Variable that is to be considered a module parameter on the space of 
        Stiefel manifold.
    """
    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()
    
"""
Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
"""
class SPDTransform(nn.Module):

    def __init__(self, input_size, output_size):
        super(SPDTransform, self).__init__()
        self.increase_dim = None
        if output_size > input_size:
            self.increase_dim = SPDIncreaseDim(input_size, output_size)
            input_size = output_size
        self.weight = StiefelParameter(torch.FloatTensor(input_size, output_size), requires_grad=True)
        nn.init.orthogonal_(self.weight)

    def forward(self, input):
        output = input
        if self.increase_dim:
            output = self.increase_dim(output)
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(input.size(0), -1, -1)
        output = torch.bmm(weight.transpose(1,2), torch.bmm(output, weight))

        return output


class SPDIncreaseDim(nn.Module):

    def __init__(self, input_size, output_size):
        super(SPDIncreaseDim, self).__init__()
        self.register_buffer('eye', torch.eye(output_size, input_size))
        add = np.asarray([0] * input_size + [1] * (output_size-input_size), dtype=np.float32)
        self.register_buffer('add', torch.from_numpy(np.diag(add)))

    def forward(self, input):
        eye = self.eye.unsqueeze(0)
        eye = eye.expand(input.size(0), -1, -1)
        add = self.add.unsqueeze(0)
        add = add.expand(input.size(0), -1, -1)

        output = torch.baddbmm(add, eye, torch.bmm(input, eye.transpose(1,2)))

        return output

"""
Yu, K., & Salzmann, M. (2017). Second-order convolutional neural networks. arXiv preprint arXiv:1703.06817.
"""
class ParametricVectorize(nn.Module):

    def __init__(self, input_size, output_size):
        super(ParametricVectorize, self).__init__()
        self.weight = nn.Parameter(torch.ones(output_size, input_size), requires_grad=True)

    def forward(self, input):
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(input.size(0), -1, -1)
        output = torch.bmm(weight, input)
        output = torch.bmm(output, weight.transpose(1,2))
        output = torch.mean(output, 2)
        return output

"""
Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
"""
class SPDVectorize(nn.Module):

    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output

class SPDUnVectorizeFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        n = int(-.5 + 0.5 * np.sqrt(1 + 8 * input.size(1)))
        output = input.new(len(input), n, n)
        output.fill_(0)
        mask_upper = np.triu_indices(n)
        mask_diag = np.diag_indices(n)
        for k, x in enumerate(input):
            output[k][mask_upper] = x
            output[k] = output[k] + output[k].t()   
            output[k][mask_diag] /= 2
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            n = int(-.5 + 0.5 * np.sqrt(1 + 8 * input.size(1)))
            grad_input = input.new(len(input), input.size(1))
            mask = np.triu_indices(n)
            for k, g in enumerate(grad_output):
                grad_input[k] = g[mask]

        return grad_input


class SPDUnVectorize(nn.Module):

    def __init__(self):
        super(SPDUnVectorize, self).__init__()

    def forward(self, input):
        return SPDUnVectorizeFunction.apply(input)


"""
Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
"""
class SPDTangentSpaceFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        
        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s.log_()
            output[k] = u.mm(s.diag().mm(u.t()))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1); eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(1))
            for k, g in enumerate(grad_output):
                x = input[k]
                u, s, v = x.svd()
                
                g = symmetric(g)
                
                s_log_diag = s.log().diag()
                s_inv_diag = (1/s).diag()
                
                dLdV = 2*(g.mm(u.mm(s_log_diag)))
                dLdS = eye * (s_inv_diag.mm(u.t().mm(g.mm(u))))
                
                P = s.unsqueeze(1)
                P = P.expand(-1, P.size(0))
                P = P - P.t()
                mask_zero = torch.abs(P) == 0
                P = 1 / P
                P[mask_zero] = 0
                
                grad_input[k] = u.mm(symmetric(P.t() * (u.t().mm(dLdV)))+dLdS).mm(u.t())


        return grad_input


"""
Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
"""
class SPDTangentSpace(nn.Module):

    def __init__(self, input_size, vectorize=True):
        super(SPDTangentSpace, self).__init__()
        self.vectorize = vectorize
        if vectorize:
            self.vec = SPDVectorize(input_size)

    def forward(self, input):
        output = SPDTangentSpaceFunction.apply(input)
        if self.vectorize:
            output = self.vec(output)

        return output


class SPDUnTangentSpaceFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        
        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s.exp_()
            output[k] = u.mm(s.diag().mm(u.t()))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1); eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(1))
            for k, g in enumerate(grad_output):
                x = input[k]
                u, s, v = x.svd()

                g = symmetric(g)
                
                s_exp_diag = s.exp().diag()
                
                dLdV = 2*(g.mm(u.mm(s_exp_diag)))
                dLdS = eye * (s_exp_diag.mm(u.t().mm(g.mm(u))))
                
                P = s.unsqueeze(1)
                P = P.expand(-1, P.size(0))
                P = P - P.t()
                mask_zero = torch.abs(P) == 0
                P = 1 / P
                P[mask_zero] = 0
                
                grad_input[k] = u.mm(symmetric(P.t() * (u.t().mm(dLdV)))+dLdS).mm(u.t())


        return grad_input


class SPDUnTangentSpace(nn.Module):

    def __init__(self, unvectorize=True):
        super(SPDUnTangentSpace, self).__init__()
        self.unvectorize = unvectorize
        if unvectorize:
            self.unvec = SPDUnVectorize()

    def forward(self, input):
        if self.unvectorize:
            input = self.unvec(input)
        output = SPDUnTangentSpaceFunction.apply(input)
        return output


"""
Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
"""
class SPDRectifiedFunction(Function):

    @staticmethod
    def forward(ctx, input, epsilon):
        ctx.save_for_backward(input, epsilon)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s[s < epsilon[0]] = epsilon[0]
            output[k] = u.mm(s.diag().mm(u.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, epsilon = ctx.saved_variables
        grad_input = None
        
        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1); eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(2))
            for k, g in enumerate(grad_output):
                if len(g.shape) == 1:
                    continue

                g = symmetric(g)

                x = input[k]
                u, s, v = x.svd()
                
                max_mask = s > epsilon
                s_max_diag = s.clone(); s_max_diag[~max_mask] = epsilon; s_max_diag = s_max_diag.diag()
                Q = max_mask.float().diag()
                
                dLdV = 2*(g.mm(u.mm(s_max_diag)))
                dLdS = eye * (Q.mm(u.t().mm(g.mm(u))))
                
                P = s.unsqueeze(1)
                P = P.expand(-1, P.size(0))
                P = P - P.t()
                mask_zero = torch.abs(P) == 0
                P = 1 / P
                P[mask_zero] = 0

                grad_input[k] = u.mm(symmetric(P.t() * u.t().mm(dLdV))+dLdS).mm(u.t())
            
        return grad_input, None


"""
Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
"""
class SPDRectified(nn.Module):

    def __init__(self, epsilon=1e-4):
        super(SPDRectified, self).__init__()
        self.register_buffer('epsilon', torch.FloatTensor([epsilon]))

    def forward(self, input):
        output = SPDRectifiedFunction.apply(input, self.epsilon)
        return output


class SPDPowerFunction(Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s = torch.exp(weight * torch.log(s))
            output[k] = u.mm(s.diag().mm(u.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_variables
        grad_input = None
        grad_weight = None

        eye = input.new(input.size(1))
        eye.fill_(1); eye = eye.diag()
        grad_input = input.new(input.size(0), input.size(1), input.size(2))
        grad_weight = weight.new(input.size(0), weight.size(0))
        for k, g in enumerate(grad_output):
            if len(g.shape) == 1:
                continue

            x = input[k]
            u, s, v = x.svd() 

            g = symmetric(g)
            
            s_log = torch.log(s)
            s_power = torch.exp(weight * s_log)

            s_power = s_power.diag()
            s_power_w_1 = weight * torch.exp((weight-1) * s_log)
            s_power_w_1 = s_power_w_1.diag()
            s_log = s_log.diag()
            
            grad_w = s_log.mm(u.t().mm(s_power.mm(u))).mm(g)
            grad_weight[k] = grad_w.diag()

            dLdV = 2*(g.mm(u.mm(s_power)))
            dLdS = eye * (s_power_w_1.mm(u.t().mm(g.mm(u))))
            
            P = s.unsqueeze(1)
            P = P.expand(-1, P.size(0))
            P = P - P.t()
            mask_zero = torch.abs(P) == 0
            P = 1 / P
            P[mask_zero] = 0            
            
            grad_input[k] = u.mm(symmetric(P.t() * u.t().mm(dLdV))+dLdS).mm(u.t())
                
        grad_weight = grad_weight.mean(0)
        
        return grad_input, grad_weight


class SPDPower(nn.Module):

    def __init__(self, input_dim):
        super(SPDPower, self).__init__()
        self.weight = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        # nn.init.normal(self.weight)

    def forward(self, input):
        output = SPDPowerFunction.apply(input, self.weight)
        return output


def symmetric(A):
    return 0.5 * (A + A.t())

def is_nan_or_inf(A):
    C1 = torch.nonzero(A == float('inf'))
    C2 = torch.nonzero(A != A)
    if len(C1.size()) > 0 or len(C2.size()) > 0:
        return True
    return False

def is_pos_def(x):
    x = x.cpu().numpy()
    return np.all(np.linalg.eigvals(x) > 0)

def matrix_operator(A, operator):
    u, s, v = A.svd()
    if operator == 'sqrtm':
        s.sqrt_()
    elif operator == 'rsqrtm':
        s.rsqrt_()
    elif operator == 'logm':
        s.log_()
    elif operator == 'expm':
        s.exp_()
    else:
        raise('operator %s is not implemented' % operator)
    
    output = u.mm(s.diag().mm(u.t()))
    
    return output

def tangent_space(A, ref, inverse_transform=False):
    ref_sqrt = matrix_operator(ref, 'sqrtm')
    ref_sqrt_inv = matrix_operator(ref, 'rsqrtm')
    middle = ref_sqrt_inv.mm(A.mm(ref_sqrt_inv))
    if inverse_transform:
        middle = matrix_operator(middle, 'logm')
    else:
        middle = matrix_operator(middle, 'expm')
    out = ref_sqrt.mm(middle.mm(ref_sqrt))
    return out

def untangent_space(A, ref):
    return tangent_space(A, ref, True)

def parallel_transform(A, ref1, ref2):
    print(A.size(), ref1.size(), ref2.size())
    out = untangent_space(A, ref1)
    out = tangent_space(out, ref2)
    return out

def orthogonal_projection(A, B):
    out = A - B.mm(symmetric(B.transpose(0,1).mm(A)))
    return out

def retraction(A, ref):
    data = A + ref
    # Q, R = data.qr()
    Q, R = torch.linalg.qr(data)
    # To avoid (any possible) negative values in the output matrix, we multiply the negative values by -1
    sign = (R.diag().sign() + 0.5).sign().diag()
    out = Q.mm(sign)
    return out


class StiefelMetaOptimizer(object):
    """This is a meta optimizer which uses other optimizers for updating parameters
        and remap all StiefelParameter parameters to Stiefel space after they have been updated.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.state = {}

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    if id(p) not in self.state:
                        self.state[id(p)] = p.data.clone()
                    else:
                        self.state[id(p)].fill_(0).add_(p.data)
                    
                    p.data.fill_(0)
                    trans = orthogonal_projection(p.grad.data, p.data)
                    p.grad.data.fill_(0).add_(trans)
                    
        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    trans = retraction(p.data, self.state[id(p)])
                    p.data.fill_(0).add_(trans)

        return loss

class Normalize(nn.Module):
    def __init__(self, p=2, dim=-1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, input):
        output = torch.nn.functional.normalize(input, p=self.p, dim=self.dim)
        return output