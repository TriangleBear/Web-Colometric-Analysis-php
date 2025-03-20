import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES  # Change to direct import since we're already in the Train directory

class DoubleConv(nn.Module):
    """Memory-optimized double convolution block with improved efficiency."""
    def __init__(self, in_channels, out_channels, mid_channels=None, efficient=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # Use groups=1 for more efficient memory usage
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels, momentum=0.01, eps=1e-5),  # Reduced momentum for stability
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-5),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if (bilinear):
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False, dropout_prob=0.5):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # OPTIMIZATION: Reduce feature dimensions significantly for mobile GPU
        factor = 2 if bilinear else 1
        
        # Reduced-size encoder path (32-64-128-256-512 instead of 64-128-256-512-1024)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512 // factor)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        # Reduced-size decoder path
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        
        # Output layer
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Initialize weights to avoid exploding losses
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Store features from encoder path for skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply dropout at the bottleneck
        x5 = self.dropout(x5)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        # Clear x4 from memory
        del x4
        
        x = self.up2(x, x3)
        # Clear x3 from memory
        del x3
        
        x = self.up3(x, x2)
        # Clear x2 from memory
        del x2
        
        x = self.up4(x, x1)
        # Clear x1 from memory
        del x1
        
        # Final output layer
        x = self.outc(x)
        
        return x

class YOLOHead(nn.Module):
    """Memory-optimized YOLO-style detection head."""
    def __init__(self, in_channels, num_classes):
        super(YOLOHead, self).__init__()
        # OPTIMIZATION: Reduce intermediate channels (1.5x instead of 2x)
        mid_channels = int(in_channels * 1.5)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels, momentum=0.01),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1)
        )
        
        # Initialize weights with Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.conv(x)
        return features

class UNetYOLO(nn.Module):  
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):  
        """Memory-optimized model for RTX 4050 mobile.""" 
        super(UNetYOLO, self).__init__()
        self.unet = UNet(in_channels, 32, bilinear=False, dropout_prob=dropout_prob)  # Reduced to 32 features
        
        # Expose decoder layers
        self.decoder = nn.ModuleList([self.unet.up1, self.unet.up2, self.unet.up3, self.unet.up4])
        
        # Add batch normalization after UNet for better stability
        self.bn = nn.BatchNorm2d(32, momentum=0.01)
        
        # YOLO head with memory optimization
        self.yolo_head = YOLOHead(32, out_channels)
        
        # Memory-efficient attention module
        self.class_attention = ClassAttentionModule(32, out_classes=min(4, out_channels))  # Reduce attention classes

        # Streamlined auxiliary classifier
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 64),  # Reduced hidden dimension
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, out_channels)
        )
        
        # Specialized heads with reduced parameters
        self.specialized_heads = nn.ModuleDict({
            'strip': nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),  # Reduced channels
                nn.BatchNorm2d(32, momentum=0.01),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1)
            ),
            'background': nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),  # Reduced channels
                nn.BatchNorm2d(32, momentum=0.01),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1)
            )
        })

        self._init_weights()
        
        # Enable gradient checkpointing to save memory
        self.use_checkpointing = True

    def _init_weights(self):
        # Initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Memory-efficient forward pass with explicit cleanup
        if self.use_checkpointing and self.training:
            # Use gradient checkpointing during training to save memory
            features = torch.utils.checkpoint.checkpoint(self.unet, x)
        else:
            features = self.unet(x)
        
        features = self.bn(features)
        
        # Apply class-specific attention
        attended_features = self.class_attention(features)
        
        # Get segmentation map from YOLO head
        segmentation_map = self.yolo_head(attended_features)
        
        # Get specialized outputs with memory-efficient approach
        if self.training:
            strip_output = self.specialized_heads['strip'](features)
            bg_output = self.specialized_heads['background'](features)
            
            # Replace the corresponding channels in segmentation map
            segmentation_map[:, 11:12, :, :] = strip_output  # Class 11 is Strip
            segmentation_map[:, 9:10, :, :] = bg_output      # Class 9 is Background
            
            # Get auxiliary classification output
            aux_output = self.aux_classifier(features)
            
            # Clean up memory
            del attended_features, features
            torch.cuda.empty_cache()
            
            return segmentation_map, aux_output, strip_output, bg_output
        else:
            # Memory-efficient inference
            strip_output = self.specialized_heads['strip'](features)
            bg_output = self.specialized_heads['background'](features)
            
            # Apply outputs directly
            segmentation_map[:, 11:12, :, :] = strip_output  # Class 11 is Strip
            segmentation_map[:, 9:10, :, :] = bg_output      # Class 9 is Background
            
            # Clean up memory
            del attended_features, features, strip_output, bg_output
            torch.cuda.empty_cache()
            
            return segmentation_map

# NEW: Add a class attention module to focus on underrepresented classes
class ClassAttentionModule(nn.Module):
    """Memory-efficient class attention module."""
    def __init__(self, in_channels, num_classes):
        super(ClassAttentionModule, self).__init__()
        # OPTIMIZATION: Reduce intermediate channels
        reduced_channels = max(8, in_channels // 8)
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Generate class-specific attention weights
        channel_weights = self.channel_attention(x)
        
        # Apply attention mechanism
        attended_features = x * channel_weights
        
        return attended_features

# Add a more memory-efficient LiteUNet for mobile GPU
class LiteUNet(nn.Module):
    """Ultra-lightweight UNet for RTX 4050 mobile."""
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super(LiteUNet, self).__init__()
        
        # Even more reduced feature dimensions
        self.inc = DoubleConv(in_channels, 24)
        self.down1 = Down(24, 48)
        self.down2 = Down(48, 96)
        self.down3 = Down(96, 192)
        
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        self.up1 = Up(192, 96)
        self.up2 = Up(96, 48)
        self.up3 = Up(48, 24)
        
        self.outc = nn.Conv2d(24, out_channels, kernel_size=1)
        
        self._init_weights()
        
        # Enable gradient checkpointing
        self.use_checkpointing = True
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Memory-efficient forward pass with explicit cleanup
        if self.use_checkpointing and self.training:
            # Use gradient checkpointing during training
            x1 = self.inc(x)
            x2 = torch.utils.checkpoint.checkpoint(self.down1, x1)
            x3 = torch.utils.checkpoint.checkpoint(self.down2, x2)
            x4 = torch.utils.checkpoint.checkpoint(self.down3, x3)
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
        
        x4 = self.dropout(x4)
        
        x = self.up1(x4, x3)
        del x3, x4
        torch.cuda.empty_cache()
        
        x = self.up2(x, x2)
        del x2
        torch.cuda.empty_cache()
        
        x = self.up3(x, x1)
        del x1
        torch.cuda.empty_cache()
        
        x = self.outc(x)
        
        return x
