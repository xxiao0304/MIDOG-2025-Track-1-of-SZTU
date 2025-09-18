import torch
import torch.nn as nn
from torch.autograd import Function
import math

import torch
import torch.nn as nn
from torch.autograd import Function
import math
from torch.cuda.amp import custom_fwd, custom_bwd

import torch
import torch.nn as nn
from torch.autograd import Function
import math
from torch.cuda.amp import custom_fwd, custom_bwd

class SkaFn(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # 解析参数
        ks = int(math.sqrt(w.shape[2]))  # 核大小 (K×K)
        pad = (ks - 1) // 2              # 填充大小
        n, ic, h, wd = x.shape           # x形状: (N, C_in, H, W)
        _, wc, _, _, _ = w.shape         # w形状: (N, C_w, K², H, W)
        G = ic // wc                     # 分组数 (G = C_in / C_w)
        
        # 保存反向传播所需参数
        ctx.ks = ks
        ctx.pad = pad
        ctx.G = G
        ctx.save_for_backward(x, w)
        
        # 输入填充与滑动窗口提取
        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='constant', value=0.0)
        x_windows = x_padded.unfold(2, ks, 1).unfold(3, ks, 1)  # (N, C_in, H, W, K, K)
        x_windows = x_windows.permute(0, 1, 4, 5, 2, 3).contiguous()  # (N, C_in, K, K, H, W)
        x_windows = x_windows.view(n, ic, ks*ks, h, wd)  # (N, C_in, K², H, W)
        
        # 分组处理与加权聚合
        x_grouped = x_windows.view(n, G, wc, ks*ks, h, wd)  # (N, G, C_w, K², H, W)
        w_grouped = w.view(n, 1, wc, ks*ks, h, wd)          # (N, 1, C_w, K², H, W)
        out_grouped = torch.sum(x_grouped * w_grouped, dim=3)  # 沿K²维度求和
        out = out_grouped.view(n, ic, h, wd)  # 合并分组: (N, C_in, H, W)
        
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, go: torch.Tensor) -> tuple:
        ks = ctx.ks
        pad = ctx.pad
        G = ctx.G
        x, w = ctx.saved_tensors
        n, ic, h, wd = x.shape
        _, wc, k_sq, w_h, w_w = w.shape  # 解析w的维度: (N, C_w, K², H, W)
        
        # 计算x的梯度 (gx)
        gx = None
        if ctx.needs_input_grad[0]:
            # 填充梯度并提取窗口
            go_padded = torch.nn.functional.pad(go, (pad, pad, pad, pad), mode='constant', value=0.0)
            go_windows = go_padded.unfold(2, ks, 1).unfold(3, ks, 1)  # (N, C_in, H, W, K, K)
            go_windows = go_windows.permute(0, 1, 4, 5, 2, 3).contiguous()  # (N, C_in, K, K, H, W)
            go_windows = go_windows.view(n, ic, ks*ks, h, wd)  # (N, C_in, K², H, W)
            
            # 分组处理并求和
            go_grouped = go_windows.view(n, G, wc, ks*ks, h, wd)  # (N, G, C_w, K², H, W)
            w_grouped = w.view(n, 1, wc, ks*ks, h, wd)            # (N, 1, C_w, K², H, W)
            gx_grouped = torch.sum(go_grouped * w_grouped, dim=3)  # 沿K²维度求和
            gx = gx_grouped.view(n, ic, h, wd)  # 合并分组: (N, C_in, H, W)
        
        # 计算w的梯度 (gw) - 核心修正
        gw = None
        if ctx.needs_input_grad[1]:
            # 填充输入x并提取窗口
            x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='constant', value=0.0)
            x_windows = x_padded.unfold(2, ks, 1).unfold(3, ks, 1)  # (N, C_in, H, W, K, K)
            x_windows = x_windows.permute(0, 1, 4, 5, 2, 3).contiguous()  # (N, C_in, K, K, H, W)
            x_windows = x_windows.view(n, ic, ks*ks, h, wd)  # (N, C_in, K², H, W)
            
            # 分组处理（严格基于w的维度）
            x_grouped = x_windows.view(n, G, wc, ks*ks, h, wd)  # (N, G, C_w, K², H, W)
            go_grouped = go.view(n, G, wc, 1, h, wd)  # (N, G, C_w, 1, H, W) - 扩展K²维度
            
            # 计算分组梯度并聚合（关键：对G维度求和）
            gw_grouped = x_grouped * go_grouped  # (N, G, C_w, K², H, W)
            gw = gw_grouped.sum(dim=1)  # 聚合分组维度G: (N, C_w, K², H, W)
            
            # 强制形状匹配（应对极端情况）
            if gw.shape != w.shape:
                gw = gw[:, :wc, :k_sq, :w_h, :w_w].contiguous()
        
        return gx, gw

class SKA(torch.nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return SkaFn.apply(x, w)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)

        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        # 确保w的形状正确：(batch, C_w, K², H, W)
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w

class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        w = self.lkp(x)
        # 调试：打印w的形状
        # print(f"LSConv forward: w.shape={w.shape}")
        return self.bn(self.ska(x, w)) + x

# 以下是其他辅助类和主函数，保持不变
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class Bottleneck_LSConv(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = LSConv(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_LSConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_LSConv(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    dim = 64
    lsconv = LSConv(dim=dim).to(device)
    print(f"LSConv 模块初始化完成，参数总量: {sum(p.numel() for p in lsconv.parameters())}")

    batch_size = 2
    height, width = 31, 27
    x = torch.randn(batch_size, dim, height, width).to(device)
    x.requires_grad_(True)
    print(f"\n输入特征图形状: {x.shape}")

    # 前向传播测试
    lsconv.eval()
    with torch.no_grad():
        output = lsconv(x)
    print(f"输出特征图形状: {output.shape}")
    print("输出与输入形状是否一致:", output.shape == x.shape)

    # 反向传播测试
    lsconv.train()
    output = lsconv(x)
    loss = output.sum()
    loss.backward()
    
    # 检查梯度是否存在且形状正确
    if x.grad is not None:
        print(f"x的梯度形状: {x.grad.shape} (应与x一致)")
    else:
        print("x的梯度不存在!")
    
    print("模块测试完成！")

if __name__ == "__main__":
    main()
