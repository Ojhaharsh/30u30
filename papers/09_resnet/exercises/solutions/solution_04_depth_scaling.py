"""
Solution 4: Depth Scaling Comparison
====================================

Compare ResNets of different depths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BasicBlock(nn.Module):
    """Basic block for ResNet-18/34."""
    expansion = 1
    
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50+."""
    expansion = 4
    
    def __init__(self, in_ch, mid_ch, stride=1, downsample=None):
        super().__init__()
        out_ch = mid_ch * self.expansion
        
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + identity)


# ResNet configurations
CONFIGS = {
    18:  (BasicBlock, [2, 2, 2, 2]),
    34:  (BasicBlock, [3, 4, 6, 3]),
    50:  (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3]),
}


def compare_resnet_depths():
    """Compare different ResNet depths."""
    print("ResNet Depth Comparison")
    print("=" * 60)
    print(f"{'Depth':<8} {'Block Type':<12} {'Params (M)':<12} {'FLOPs (G)':<12}")
    print("-" * 60)
    
    for depth in [18, 34, 50, 101, 152]:
        block, layers = CONFIGS[depth]
        
        # Use torchvision for accurate counts
        model_fn = getattr(models, f'resnet{depth}')
        model = model_fn(pretrained=False)
        
        params = sum(p.numel() for p in model.parameters()) / 1e6
        
        # Estimate FLOPs (approximate)
        x = torch.randn(1, 3, 224, 224)
        flops = estimate_flops(model, x) / 1e9
        
        block_name = 'Basic' if block == BasicBlock else 'Bottleneck'
        print(f"{depth:<8} {block_name:<12} {params:<12.1f} {flops:<12.1f}")
    
    print("-" * 60)
    print("\nObservations:")
    print("  - ResNet-18/34 use BasicBlock (2 convs per block)")
    print("  - ResNet-50+ use Bottleneck (3 convs per block)")
    print("  - Bottleneck is more parameter efficient at high depth")
    print("  - Diminishing returns above ~100 layers for ImageNet")


def estimate_flops(model, x):
    """Simple FLOPs estimate based on conv layer sizes."""
    total = 0
    
    def hook(module, inp, out):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            # FLOPs ≈ 2 * K^2 * C_in * C_out * H_out * W_out
            out_h, out_w = out.shape[2:]
            k = module.kernel_size[0]
            total += 2 * k * k * module.in_channels * module.out_channels * out_h * out_w
    
    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(hook))
    
    with torch.no_grad():
        model(x)
    
    for h in handles:
        h.remove()
    
    return total


if __name__ == "__main__":
    compare_resnet_depths()
