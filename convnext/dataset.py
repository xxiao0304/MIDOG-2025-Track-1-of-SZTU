import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import RandAugment  # 新增：导入RandAugment

# 仅保留两类：0=有丝分裂（mitosis），1=背景（background）
class_names = ["mitosis", "background"]

# 固定随机种子（保证可复现）
torch.manual_seed(42)


class BBoxDataset(Dataset):
    """
    边界框分类数据集
    适配目录结构：
    - 图像：datasets_fu/images/[split]/ （split: train/val/test）
    - 标签：data_class/[split]/ （txt文件，YOLO格式）
    """
    def __init__(self, split, images_root="datasets_fu/images", labels_root="data_class", transforms=None):
        self.split = split  # train/val/test
        self.images_dir = os.path.join(images_root, split)
        self.labels_dir = os.path.join(labels_root, split)
        self.transforms = transforms
        self.img_size = 512  # 原始图像尺寸（与标签匹配）
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        # 遍历所有标签文件
        txt_files = [f for f in os.listdir(self.labels_dir) if f.endswith('.txt')]
        for txt_file in txt_files:
            img_id = os.path.splitext(txt_file)[0]
            img_file = f"{img_id}.tiff"
            img_path = os.path.join(self.images_dir, img_file)
            
            if not os.path.exists(img_path):
                print(f"警告：图像不存在 - {img_path}，已跳过")
                continue  # 跳过缺失的图像
            
            # 读取标签文件
            txt_path = os.path.join(self.labels_dir, txt_file)
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            # 解析标签，仅保留0和1类
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue  # 跳过格式错误的行
                class_id = int(parts[0])
                if class_id not in [0, 1]:
                    continue  # 过滤非目标类别
                
                # 转换YOLO归一化坐标为像素坐标
                x_center = float(parts[1]) * self.img_size
                y_center = float(parts[2]) * self.img_size
                width = float(parts[3]) * self.img_size
                height = float(parts[4]) * self.img_size
                
                # 计算裁剪坐标（防止越界）
                x1 = max(0, int(x_center - width / 2))
                y1 = max(0, int(y_center - height / 2))
                x2 = min(self.img_size, int(x_center + width / 2))
                y2 = min(self.img_size, int(y_center + height / 2))
                
                # 新增：过滤过小样本（避免增强后细胞特征丢失）
                if (x2 - x1) < 8 or (y2 - y1) < 8:
                    print(f"警告：过小边界框（宽={(x2-x1)}, 高={(y2-y1)}）- {img_path}，已跳过")
                    continue
                
                samples.append({
                    'img_path': img_path,
                    'class_id': class_id,
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 读取并裁剪图像（保留原始边界框裁剪逻辑）
        try:
            image = Image.open(sample['img_path']).convert('RGB')
            bbox_image = image.crop((sample['x1'], sample['y1'], sample['x2'], sample['y2']))
        except Exception as e:
            print(f"警告：图像裁剪失败 - {sample['img_path']}，错误：{e}，返回空样本")
            # 返回空白图像和无效标签（避免训练中断，后续可过滤）
            bbox_image = Image.new('RGB', (32, 32), color=0)
            if self.transforms:
                bbox_image = self.transforms(bbox_image)
            return bbox_image, -1  # 用-1标记无效样本
        
        # 应用数据变换（含增强策略）
        if self.transforms:
            bbox_image = self.transforms(bbox_image)
        
        return bbox_image, sample['class_id']


def get_transforms(is_train=True):
    """获取数据变换（训练集：RandAugment + Random Erasing + 原有增强；测试集：仅基础处理）"""
    # 基础变换（所有集共享：Resize + ToTensor + Normalize）
    base_transforms = [
        transforms.Resize((64, 64)),  # 统一缩放到64x64（适配模型输入）
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    if is_train:
        # 训练集增强流程：几何/颜色增强 → RandAugment → 基础变换 → Random Erasing
        train_transforms = [
            # 1. 原有基础增强（保留并调整参数，适配细胞样本）
            transforms.RandomHorizontalFlip(p=0.5),  # 新增：水平翻转（病理图像左右对称无意义，概率0.5）
            transforms.RandomVerticalFlip(p=0.5),    # 新增：垂直翻转（同理）
            transforms.RandomRotation(degrees=15),   # 旋转角度从10°调整为15°，增加多样性
            transforms.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.1  # 新增饱和度调整，避免过度改变细胞染色特征
            ),
            # 2. RandAugment：自动组合3种增强操作，强度5（适配细胞级样本，避免失真）
            RandAugment(num_ops=3, magnitude=5),
            # 3. 基础变换（Resize → ToTensor → Normalize）
            *base_transforms,
            # 4. Random Erasing：随机擦除冗余区域（不破坏细胞核心）
            transforms.RandomErasing(
                p=0.5,                # 50%概率执行擦除
                scale=(0.02, 0.15),   # 擦除面积：2%-15%（细胞样本小，擦除比例不宜过大）
                ratio=(0.3, 3.3),     # 擦除区域宽高比：0.3-3.3（覆盖长条/方形区域）
                value=0.5             # 擦除填充值：0.5（接近病理图像背景灰度，避免突兀）
            )
        ]
        return transforms.Compose(train_transforms)
    else:
        # 验证/测试集：仅基础变换（无增强，保证评估一致性）
        return transforms.Compose(base_transforms)


def create_datasets():
    """创建数据集（直接使用已有的train/val/test划分）"""
    # 训练集变换（带RandAugment + Random Erasing）
    train_dataset = BBoxDataset(
        split="train",
        transforms=get_transforms(is_train=True)
    )
    
    # 验证集变换（无增强）
    val_dataset = BBoxDataset(
        split="val",
        transforms=get_transforms(is_train=False)
    )
    
    # 测试集变换（无增强）
    test_dataset = BBoxDataset(
        split="test",
        transforms=get_transforms(is_train=False)
    )
    
    # 打印数据集信息（新增无效样本统计）
    def print_dataset_info(ds, split_name):
        print(f"\n{split_name}集统计:")
        print(f"  有效样本总数: {len(ds)}")
        # 统计类别分布
        class_counts = {0:0, 1:0}
        for sample in ds.samples:
            class_counts[sample['class_id']] += 1
        print(f"  类别分布: {class_names[0]}={class_counts[0]} ({class_counts[0]/len(ds)*100:.1f}%), "
              f"{class_names[1]}={class_counts[1]} ({class_counts[1]/len(ds)*100:.1f}%)")
        # 统计边界框尺寸范围（新增，便于评估样本质量）
        widths = [sample['x2'] - sample['x1'] for sample in ds.samples]
        heights = [sample['y2'] - sample['y1'] for sample in ds.samples]
        print(f"  边界框尺寸: 宽={min(widths)}-{max(widths)}px, 高={min(heights)}-{max(heights)}px")
    
    print_dataset_info(train_dataset, "训练")
    print_dataset_info(val_dataset, "验证")
    print_dataset_info(test_dataset, "测试")
    
    return train_dataset, val_dataset, test_dataset