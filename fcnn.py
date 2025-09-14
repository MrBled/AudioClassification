import torch
import torch.nn as nn
import torch.nn.functional as F



def kaiming_init(module: nn.Module):
    """Kaiming (He) init for convs; BN weight=1, bias=0."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, pad=0, learn_bn=True, use_relu=True):
        super().__init__()
        self.pad = nn.ZeroPad2d(pad) if pad else nn.Identity()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, affine=learn_bn)
        self.use_relu = use_relu
    def forward(self, x):
        x = self.pad(x); x = self.conv(x); x = self.bn(x)
        return F.relu(x, inplace=True) if self.use_relu else x

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=2):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
    def forward(self, x):
        w = self.gap(x)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w

class ConvLayer1Abs(nn.Module):
    def __init__(self, in_ch, out_ch, learn_bn=True):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(in_ch, affine=learn_bn)
        self.conv1 = ConvBNReLU(in_ch, out_ch, k=5, s=2, pad=2, learn_bn=learn_bn, use_relu=True)
        self.conv2 = ConvBNReLU(out_ch, out_ch, k=3, s=1, pad=1, learn_bn=learn_bn, use_relu=True)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.bn0(x); x = self.conv1(x); x = self.conv2(x); return self.pool(x)

class ConvLayer2Abs(nn.Module):
    def __init__(self, in_ch, out_ch, learn_bn=True):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, k=3, s=1, pad=1, learn_bn=learn_bn, use_relu=True)
        self.conv2 = ConvBNReLU(out_ch, out_ch, k=3, s=1, pad=1, learn_bn=learn_bn, use_relu=True)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); return self.pool(x)

class ConvLayer3Abs(nn.Module):
    def __init__(self, in_ch, out_ch, learn_bn=True, p_drop=0.3):
        super().__init__()
        self.c1 = ConvBNReLU(in_ch,  out_ch, k=3, s=1, pad=1, learn_bn=learn_bn, use_relu=True)
        self.c2 = ConvBNReLU(out_ch, out_ch, k=3, s=1, pad=1, learn_bn=learn_bn, use_relu=True)
        self.c3 = ConvBNReLU(out_ch, out_ch, k=3, s=1, pad=1, learn_bn=learn_bn, use_relu=True)
        self.c4 = ConvBNReLU(out_ch, out_ch, k=3, s=1, pad=1, learn_bn=learn_bn, use_relu=True)
        self.do = nn.Dropout(p_drop)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.c1(x); x = self.do(x)
        x = self.c2(x); x = self.do(x)
        x = self.c3(x); x = self.do(x)
        x = self.c4(x); x = self.pool(x)
        return x

class ResNetLayer1x1(nn.Module):
    """TF resnet_layer with kernel_size=1, BN(affine=learn_bn), optional ReLU."""
    def __init__(self, in_ch: int, out_ch: int, learn_bn: bool = True, use_relu: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, affine=learn_bn)
        self.use_relu = use_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True) if self.use_relu else x

class FCNN(nn.Module):
    def __init__(self, num_classes, in_channels=3,
                 block_channels=(144, 288, 576),  # <- matches (24,48,96)*6
                 attention_ratio=2, return_probs=False):
        super().__init__()
        b1, b2, b3 = block_channels
        self.block1 = ConvLayer1Abs(in_ch=in_channels, out_ch=b1, learn_bn=True)
        self.block2 = ConvLayer2Abs(in_ch=b1,          out_ch=b2, learn_bn=True)
        self.block3 = ConvLayer3Abs(in_ch=b2,          out_ch=b3, learn_bn=True, p_drop=0.3)

        self.out_conv = nn.Sequential(
            nn.Conv2d(b3, num_classes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_classes, affine=False),
            nn.ReLU(inplace=True),
        )
        self.att = ChannelAttention(num_classes, ratio=attention_ratio)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.return_probs = return_probs

        # simple init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.out_conv(x)
        x = self.att(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)   # (B, num_classes)
        return F.softmax(x, dim=1) if self.return_probs else x


if __name__ == "__main__":
    # Suppose your TF input was [batch, time, freq, channels] = [B, T, 128, 6]
    B, T, F, C = 2, 500, 128, 6
    x_tf_like = torch.randn(B, T, F, C)  # channels-last
    model = FCNN(num_classes=10, in_channels=6, num_filters=(24, 48, 96), attention_ratio=2, return_probs=False)
    logits = model(x_tf_like)            # the model will accept channels-last and permute
    print(logits.shape)                  # -> [2, 10]
