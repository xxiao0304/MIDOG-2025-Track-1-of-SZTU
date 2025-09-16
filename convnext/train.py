import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import logging
import numpy as np  # 新增：用于Mixup/CutMix的随机参数生成
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.console import Console
import torch.nn.functional as F

# 导入自定义模块
from dataset import create_datasets, class_names  # 数据集和类别（已含RandAugment+Random Erasing）
from model import MitosisClassifier  # 自定义模型（无预训练）
from utils import (
    create_output_dirs, 
    save_metrics,    # 指标保存（含混淆矩阵）
    save_checkpoint, # 模型保存
    evaluate_model   # 模型评估
)

# 设置可见GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

def setup_logging(log_dir):
    """初始化日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{time.strftime("%Y%m%d-%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# -------------------------- 修复：Mixup/CutMix 核心实现 --------------------------
def mixup_data(x, y, alpha=0.2, device=None):
    """Mixup：混合两个样本（修复lam的格式问题）"""
    if alpha <= 0 or x.size(0) < 2:
        return x, y, y, 1.0  # 1.0 是float，无格式问题
    
    lam = np.random.beta(alpha, alpha)  # 此时lam是numpy.float64
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    # 无需额外转换：np.random.beta返回的是numpy数值，可直接用.item()转为float（或直接使用）
    return mixed_x, y_a, y_b, float(lam)  # 显式转为float，彻底避免格式问题


def cutmix_data(x, y, alpha=0.2, device=None):
    """CutMix：裁剪粘贴样本（核心修复：lam从张量转float）"""
    if alpha <= 0 or x.size(0) < 2:
        return x, y, y, 1.0  # 1.0 是float
    
    lam = np.random.beta(alpha, alpha)  # 初始为numpy.float64
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size, device=device)
    
    # 计算裁剪区域（原有逻辑不变）
    cut_rat = np.sqrt(1. - lam)
    cut_h = int(h * cut_rat)
    cut_w = int(w * cut_rat)
    
    cx = torch.randint(0, w, (batch_size,), device=device)
    cy = torch.randint(0, h, (batch_size,), device=device)
    
    bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
    bby1 = torch.clamp(cy - cut_h // 2, 0, h)
    bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
    bby2 = torch.clamp(cy + cut_h // 2, 0, h)
    
    # 裁剪粘贴（原有逻辑不变）
    for i in range(batch_size):
        x[i, :, bby1[i]:bby2[i], bbx1[i]:bbx2[i]] = x[index[i], :, bby1[i]:bby2[i], bbx1[i]:bbx2[i]]
    lam_tensor = 1 - ((bbx2 - bbx1) * (bby2 - bby1)).float() / (h * w)
    lam = lam_tensor.mean().item()  # .item() 从张量中提取Python float值
    # -----------------------------------------------------------------------------------------
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam  # 此时lam是Python float，可正常格式化


def mixup_cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """适配Mixup/CutMix的混合损失计算（兼容Focal Loss和Contrastive Loss）"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -------------------------- 2. 原有损失函数（保持不变） --------------------------
class FocalLoss(nn.Module):
    """Focal Loss：解决类别不平衡"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class ContrastiveLoss(nn.Module):
    """对比损失：增强特征区分度"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
        
        sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float() - torch.eye(batch_size, device=features.device)
        
        logits = sim_matrix / self.temperature
        exp_logits = torch.exp(logits)
        
        positive_sum = (exp_logits * mask).sum(dim=1)
        negative_sum = exp_logits.sum(dim=1) - exp_logits.diag()
        
        valid_mask = (positive_sum > 0) & (negative_sum > 0)
        if not valid_mask.any():
            return torch.tensor(0.0, device=features.device)
            
        return -torch.log(positive_sum[valid_mask] / negative_sum[valid_mask]).mean()

# -------------------------- 3. 修改：train_model 函数（适配混合策略+无效样本过滤） --------------------------
def train_model(model, train_loader, val_loader, criterion_focal, criterion_contrast,
                optimizer, num_epochs, device, logger, output_dir,
                mixup_alpha=0.2, cutmix_alpha=0.2, mix_prob=0.5):
    """
    完整训练流程（新增：Mixup/CutMix + 无效样本过滤）
    Args:
        mixup_alpha: Mixup的Beta参数（0=禁用）
        cutmix_alpha: CutMix的Beta参数（0=禁用）
        mix_prob: 每个批次使用混合策略的概率（0.5=平衡原始/混合样本）
    """
    best_val_loss = float('inf')
    best_epoch = 0
    early_stopping_counter = 0
    early_stopping_patience = num_epochs*0.10  # 早停耐心值
    
    # 余弦退火学习率调度（原有逻辑不变）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    console = Console()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        total_task = progress.add_task("[bold green]训练总进度", total=num_epochs)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            current_lr = optimizer.param_groups[0]['lr']
            
            # -------------------------- 训练阶段（核心修改） --------------------------
            model.train()
            train_total_loss = 0.0
            train_correct = 0
            train_samples = 0
            invalid_sample_count = 0  # 统计无效样本数
            
            train_task = progress.add_task("[bold blue]训练批次", total=len(train_loader))
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # 步骤1：过滤无效样本（标签=-1，来自dataset.py的裁剪失败样本）
                valid_mask = labels != -1
                if not valid_mask.any():
                    invalid_sample_count += inputs.size(0)
                    progress.update(train_task, advance=1, description=f"批次{batch_idx+1}：全为无效样本，跳过")
                    continue
                # 筛选有效样本
                inputs = inputs[valid_mask].to(device)
                labels = labels[valid_mask].to(device)
                batch_size = inputs.size(0)
                if batch_size < 2:  # 批次量<2时不使用混合策略（避免计算错误）
                    use_mix = False
                else:
                    use_mix = torch.rand(1).item() < mix_prob  # 随机决定是否混合
                
                # 步骤2：应用Mixup/CutMix（仅对有效样本）
                if use_mix:
                    # 50%概率用Mixup，50%用CutMix
                    if torch.rand(1).item() < 0.5 and mixup_alpha > 0:
                        inputs, y_a, y_b, lam = mixup_data(
                            inputs, labels, alpha=mixup_alpha, device=device
                        )
                        mix_type = "Mixup"
                    else:
                        inputs, y_a, y_b, lam = cutmix_data(
                            inputs, labels, alpha=cutmix_alpha, device=device
                        )
                        mix_type = "CutMix"
                    logger.debug(f"Epoch {epoch+1} 批次{batch_idx+1}：使用{mix_type}，混合比例={lam:.4f}")
                else:
                    y_a = y_b = labels
                    lam = 1.0
                    mix_type = "None"
                
                # 步骤3：模型前向传播与损失计算
                optimizer.zero_grad()
                outputs, features = model(inputs)  # 解包（分类输出, 特征）
                
                # 混合损失计算（Focal Loss + Contrastive Loss）
                focal_loss = mixup_cutmix_criterion(criterion_focal, outputs, y_a, y_b, lam)
                contrast_loss = mixup_cutmix_criterion(criterion_contrast, features, y_a, y_b, lam)
                total_loss = focal_loss + contrast_loss  # 等权重融合（可按需调整）
                
                # 步骤4：反向传播与参数更新
                total_loss.backward()
                optimizer.step()
                
                # 步骤5：累计训练指标（混合策略下需加权计算准确率）
                _, preds = torch.max(outputs, 1)
                correct = lam * (preds == y_a).sum().item() + (1 - lam) * (preds == y_b).sum().item()
                
                train_total_loss += total_loss.item() * batch_size
                train_correct += correct
                train_samples += batch_size
                invalid_sample_count += (valid_mask == False).sum().item()  # 累计无效样本数
                
                # 更新进度条（显示当前批次的损失和混合类型）
                progress.update(
                    train_task, 
                    advance=1, 
                    description=f"Focal: {focal_loss.item():.4f} | Contrast: {contrast_loss.item():.4f} | 混合: {mix_type}"
                )
            
            progress.remove_task(train_task)
            
            # 计算训练集指标（排除无效样本）
            if train_samples == 0:
                train_avg_loss = 0.0
                train_acc = 0.0
            else:
                train_avg_loss = train_total_loss / train_samples
                train_acc = 100. * train_correct / train_samples
            
            # -------------------------- 验证阶段（原有逻辑不变，仅过滤无效样本） --------------------------
            model.eval()
            val_total_loss = 0.0
            val_correct = 0
            val_samples = 0
            val_invalid_count = 0
            
            val_task = progress.add_task("[bold purple]验证批次", total=len(val_loader))
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # 过滤无效样本（与训练集逻辑一致）
                    valid_mask = labels != -1
                    if not valid_mask.any():
                        val_invalid_count += inputs.size(0)
                        progress.update(val_task, advance=1)
                        continue
                    inputs = inputs[valid_mask].to(device)
                    labels = labels[valid_mask].to(device)
                    batch_size = inputs.size(0)
                    
                    outputs, _ = model(inputs)  # 验证时无需特征
                    focal_loss = criterion_focal(outputs, labels)
                    val_total_loss += focal_loss.item() * batch_size
                    
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_samples += batch_size
                    val_invalid_count += (valid_mask == False).sum().item()
                    
                    progress.update(val_task, advance=1)
            
            progress.remove_task(val_task)
            
            # 计算验证集指标
            if val_samples == 0:
                val_avg_loss = float('inf')
                val_acc = 0.0
            else:
                val_avg_loss = val_total_loss / val_samples
                val_acc = 100. * val_correct / val_samples
            
            # -------------------------- 模型保存与早停（原有逻辑不变） --------------------------
            # 保存最佳模型（基于验证损失）
            if val_avg_loss < best_val_loss and val_samples > 0:
                best_val_loss = val_avg_loss
                best_epoch = epoch
                model_to_save = model.module if hasattr(model, 'module') else model
                save_checkpoint(
                    model_to_save, optimizer, epoch, val_avg_loss,
                    os.path.join(output_dir, "models", "best_model.pth")
                )
                early_stopping_counter = 0
                logger.info(f"Epoch {epoch+1}：更新最佳模型（验证损失={val_avg_loss:.4f}）")
            else:
                early_stopping_counter += 1
                logger.info(f"早停计数器: {early_stopping_counter}/{early_stopping_patience}")
            
            # 保存最后一轮模型
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint(
                model_to_save, optimizer, epoch, val_avg_loss,
                os.path.join(output_dir, "models", "last_model.pth")
            )
            
            # -------------------------- 日志输出（新增无效样本统计） --------------------------
            epoch_time = time.time() - start_time
            logger.info(
                f'Epoch {epoch+1}/{num_epochs} | 耗时: {epoch_time:.2f}s | LR: {current_lr:.6f}\n'
                f'训练集 - 有效样本: {train_samples} | 无效样本: {invalid_sample_count} | 损失: {train_avg_loss:.4f} | 准确率: {train_acc:.2f}%\n'
                f'验证集 - 有效样本: {val_samples} | 无效样本: {val_invalid_count} | 损失: {val_avg_loss:.4f} | 准确率: {val_acc:.2f}%\n'
                f'最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch+1})'
            )
            
            console.print(
                f"[bold yellow]Epoch {epoch+1} - 训练准确率: {train_acc:.2f}% | 验证准确率: {val_acc:.2f}% | "
                f"无效样本: 训练{invalid_sample_count}个 | 验证{val_invalid_count}个"
            )
            
            progress.update(total_task, advance=1)
            scheduler.step()
            
            # 早停判断
            if early_stopping_counter >= early_stopping_patience:
                console.print(f"[bold red]早停触发！在Epoch {epoch+1} 停止训练")
                break
    
    return model

# -------------------------- 4. 修改：main函数（传递混合策略参数） --------------------------
def main():
    # 初始化输出目录（原有逻辑不变）
    output_base = "mitosis_classification"
    output_dir, timestamp = create_output_dirs("mitosis_classification")
    
    # 初始化日志（原有逻辑不变）
    logger = setup_logging(os.path.join(output_dir, "logs"))
    logger.info("===== 训练流程启动 =====")
    
    # 设备配置（原有逻辑不变）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    logger.info(f"GPU数量: {torch.cuda.device_count()}")
    
    # 加载数据集（已含RandAugment+Random Erasing）
    logger.info("加载数据集...")
    train_dataset, val_dataset, test_dataset = create_datasets()
    logger.info(
        f"数据集大小 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}\n"
        f"类别：{class_names}（0={class_names[0]}, 1={class_names[1]}）"
    )
    
    # 数据加载器（原有逻辑不变）
    batch_size = 960  # 可根据GPU显存调整（如单卡显存不足，可降至480或240）
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=12, pin_memory=True, drop_last=True  # 新增drop_last=True：避免最后批次量<2导致混合策略失效
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=12, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=12, pin_memory=True
    )
    
    # 初始化模型（原有逻辑不变）
    logger.info("初始化模型...")
    model = MitosisClassifier(model_size="tiny", num_classes=len(class_names))
    model = model.to(device)
    
    # 多GPU支持（原有逻辑不变）
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"启用多GPU训练，设备数: {torch.cuda.device_count()}")
    
    # 损失函数与优化器（原有逻辑不变）
    alpha = torch.tensor([1, 2]).to(device)  # 类别权重（适配类别不平衡，0类权重更高）
    criterion_focal = FocalLoss(alpha=alpha, gamma=3.0) 
    criterion_contrast = ContrastiveLoss(temperature=0.15)
    optimizer = optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=5e-5  # 学习率3e-4，权重衰减抑制过拟合
    )
    
    # -------------------------- 关键修改：调用train_model时传递混合策略参数 --------------------------
    logger.info("开始训练...")
    model = train_model(
        model, train_loader, val_loader,
        criterion_focal, criterion_contrast,
        optimizer, num_epochs=400,  # 训练轮次（可根据早停调整）
        device=device, logger=logger, output_dir=output_dir,
        mixup_alpha=0.2,    # Mixup强度（0.2适中，避免细胞特征模糊）
        cutmix_alpha=0.2,   # CutMix强度（与Mixup一致）
        mix_prob=0.5        # 50%批次使用混合策略（平衡原始/增强样本）
    )
    
    # 评估最佳模型（原有逻辑不变，新增无效样本过滤）
    logger.info("评估最佳模型...")
    best_model_dir = os.path.join(output_dir, f"models/models_{timestamp}")
    best_model_path = os.path.join(best_model_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        logger.error(f"最佳模型文件不存在：{best_model_path}")
        return
    
    checkpoint = torch.load(best_model_path)
    # 加载模型（处理多GPU情况）
    model_to_load = MitosisClassifier(model_size="tiny", num_classes=len(class_names))
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    model_to_load = model_to_load.to(device)
    model_to_load.eval()
    
    # 验证集评估（过滤无效样本）
    logger.info("评估验证集...")
    val_metrics = evaluate_model(model_to_load, val_loader, device, class_names, filter_invalid=True)
    save_metrics(val_metrics, os.path.join(output_dir, "metrics"), prefix="val_")
    logger.info("验证集评估完成，指标已保存")
    
    # 测试集评估（过滤无效样本）
    logger.info("评估测试集...")
    test_metrics = evaluate_model(model_to_load, test_loader, device, class_names, filter_invalid=True)
    save_metrics(test_metrics, os.path.join(output_dir, "metrics"), prefix="test_")
    logger.info("测试集评估完成，指标已保存")
    
    logger.info("===== 训练与评估流程全部完成 =====")

if __name__ == "__main__":
    main()