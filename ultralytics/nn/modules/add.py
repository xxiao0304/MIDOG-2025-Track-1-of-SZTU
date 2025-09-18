import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite

# 动态卷积模块 (SKA + LKP)
import math
from torch.autograd import Function

class SkaFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # --------------------------- 参数解析 ---------------------------
        K = int(math.sqrt(w.shape[2]))  # 核大小 (K×K)
        pad = (K - 1) // 2  # 填充大小，确保输出尺寸与输入一致
        N, C_in, H, W = x.shape  # 输入形状: (N, C_in, H, W)
        _, C_w, _, _, _ = w.shape  # w形状: (N, C_w, K², H, W)，C_w为分组通道数
        
        # 保存反向传播所需参数
        ctx.K = K
        ctx.pad = pad
        ctx.save_for_backward(x, w)
        
        # --------------------------- 输入填充 ---------------------------
        # 对x进行填充，便于滑动窗口提取 (左右上下各填充pad)
        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='constant', value=0.0)
        
        # --------------------------- 提取滑动窗口 ---------------------------
        # 提取x的K×K滑动窗口，输出形状: (N, C_in, K, K, H, W)
        x_windows = x_padded.unfold(2, K, 1).unfold(3, K, 1)  # (N, C_in, H, W, K, K) → 调整维度顺序
        x_windows = x_windows.permute(0, 1, 4, 5, 2, 3).contiguous()  # (N, C_in, K, K, H, W)
        x_windows = x_windows.view(N, C_in, K*K, H, W)  # 展平K×K为K²: (N, C_in, K², H, W)
        
        # --------------------------- 分组加权聚合 ---------------------------
        # 核心逻辑：x窗口 × 权重w，按组累加 (原代码中ci % wc实现分组)
        # 1. 调整x和w的形状以支持分组操作
        x_grouped = x_windows.view(N, C_in // C_w, C_w, K*K, H, W)  # (N, G, C_w, K², H, W)，G为分组数
        w_grouped = w.view(N, 1, C_w, K*K, H, W)  # (N, 1, C_w, K², H, W)，广播分组维度
        
        # 2. 按组加权求和: (N, G, C_w, K², H, W) × (N, 1, C_w, K², H, W) → 求和后为 (N, G, C_w, H, W)
        out_grouped = torch.sum(x_grouped * w_grouped, dim=3)  # 沿K²维度求和
        
        # 3. 合并分组，恢复输出形状 (N, C_in, H, W)
        out = out_grouped.view(N, C_in, H, W)
        
        return out

    @staticmethod
    def backward(ctx, go: torch.Tensor) -> tuple:
        # --------------------------- 恢复参数 ---------------------------
        K = ctx.K
        pad = ctx.pad
        x, w = ctx.saved_tensors
        N, C_in, H, W = x.shape
        _, C_w, _, _, _ = w.shape
        G = C_in // C_w  # 分组数
        
        # --------------------------- 计算gx (x的梯度) ---------------------------
        # 逻辑：go与w加权求和 (与前向对称)
        pad_go = torch.nn.functional.pad(go, (pad, pad, pad, pad), mode='constant', value=0.0)
        go_windows = pad_go.unfold(2, K, 1).unfold(3, K, 1)
        go_windows = go_windows.permute(0, 1, 4, 5, 2, 3).contiguous().view(N, C_in, K*K, H, W)
        
        # 分组处理
        go_grouped = go_windows.view(N, G, C_w, K*K, H, W)
        w_grouped = w.view(N, 1, C_w, K*K, H, W)
        gx_grouped = torch.sum(go_grouped * w_grouped, dim=3)  # 沿K²维度求和
        gx = gx_grouped.view(N, C_in, H, W)
        
        # --------------------------- 计算gw (w的梯度) ---------------------------
        # 逻辑：x与go加权求和 (与前向对称)
        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='constant', value=0.0)
        x_windows = x_padded.unfold(2, K, 1).unfold(3, K, 1)
        x_windows = x_windows.permute(0, 1, 4, 5, 2, 3).contiguous().view(N, C_in, K*K, H, W)
        
        # 分组处理
        x_grouped = x_windows.view(N, G, C_w, K*K, H, W)  # (N, G, C_w, K², H, W)
        go_grouped = go.view(N, G, C_w, 1, H, W)  # (N, G, C_w, 1, H, W)，扩展K²维度
        
        # 计算梯度：(N, G, C_w, K², H, W) = (N, G, C_w, K², H, W) * (N, G, C_w, 1, H, W)
        gw_grouped = x_grouped * go_grouped
        
        # 合并分组并恢复w的形状 (N, C_w, K², H, W)
        gw = gw_grouped.view(N, C_in, K*K, H, W)
        
        return gx, gw

class SKA(torch.nn.Module):
    def __init__(self, dim, ks=3, groups=8):
        super().__init__()
        self.dim = dim
        self.ks = ks
        self.groups = groups
        
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        小核聚合模块前向传播
        
        参数:
            x: 输入特征图，形状为 (N, C_in, H, W)
            w: 聚合核权重，形状为 (N, C_w, K², H, W)，其中 K 为核大小，C_w 为分组通道数
        
        返回:
            聚合后的特征图，形状为 (N, C_in, H, W) (与x一致)
        """
        return SkaFn.apply(x, w)

class LKP(nn.Module):
    """大核感知模块 - 生成动态卷积权重"""
    def __init__(self, dim, lks=7, sks=3, groups=8):
        super().__init__()
        self.dim = dim
        self.groups = groups
        self.sks = sks
        
        # 大核感知分支
        self.cv1 = nn.Conv2d(dim, dim//2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(dim//2)
        self.cv2 = nn.Conv2d(dim//2, dim//2, kernel_size=lks, padding=lks//2, groups=dim//2)
        self.bn2 = nn.BatchNorm2d(dim//2)
        self.cv3 = nn.Conv2d(dim//2, dim//2, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(dim//2)
        
        # 生成动态权重
        self.cv4 = nn.Conv2d(dim//2, groups * sks * sks, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(groups * sks * sks)
        
    def forward(self, x):
        """生成用于SKA的动态权重"""
        B, C, H, W = x.size()
        
        # 大核特征提取
        feat = F.relu(self.bn1(self.cv1(x)))
        feat = F.relu(self.bn2(self.cv2(feat)))
        feat = F.relu(self.bn3(self.cv3(feat)))
        
        # 生成动态权重
        w = self.bn4(self.cv4(feat))  # [B, groups*sks*sks, H, W]
        w = w.view(B, self.groups, self.sks*self.sks, H, W)  # [B, groups, sks*sks, H, W]
        
        # 归一化权重
        w = F.softmax(w, dim=2)
        return w

class LSConv(nn.Module):
    """LS卷积 - 结合大核感知与小核动态聚合"""
    def __init__(self, dim, lks=7, sks=3, groups=8):
        super().__init__()
        self.lkp = LKP(dim, lks=lks, sks=sks, groups=groups)
        self.ska = SKA(dim, ks=sks, groups=groups)
        self.bn = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        """结合大核感知和小核动态卷积"""
        w = self.lkp(x)  # 生成动态权重
        x = self.ska(x, w)  # 应用动态卷积
        x = F.relu(self.bn(x))  # 归一化后添加ReLU激活
        return x

# 基于LSConv的瓶颈块
class LSBottleneck(nn.Module):
    """基于LSConv的瓶颈块，替换原Bottleneck"""
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1):
        super().__init__()
        # 确保输入输出通道相同（残差连接要求）
        assert c1 == c2, "LSBottleneck要求输入输出通道相同"
        
        # 使用LSConv作为核心操作
        self.ls_conv = LSConv(c1, groups=g)
        self.add = shortcut and c1 == c2  # 是否使用残差连接
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接
        return x + self.ls_conv(x) if self.add else self.ls_conv(x)