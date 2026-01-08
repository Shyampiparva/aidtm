"""
Standalone NAFNet Implementation

This module provides a standalone NAFNet implementation that doesn't depend on
BasicSR's full import system, avoiding the torchvision compatibility issues.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """Layer normalization for NAFNet."""
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SimpleGate(nn.Module):
    """Simple gating mechanism for NAFNet."""
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAF Block implementation."""
    
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c, data_format='channels_first')
        self.norm2 = LayerNorm(c, data_format='channels_first')

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class StandaloneNAFNet(nn.Module):
    """
    Standalone NAFNet implementation that doesn't depend on BasicSR.
    
    This implementation provides the same NAFNet-Width64 architecture
    but avoids the BasicSR import issues.
    """
    
    def __init__(self, img_channel=3, width=64, middle_blk_num=12, 
                 enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class StandaloneNAFNetLoader:
    """
    Standalone NAFNet model loader that bypasses BasicSR import issues.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the standalone NAFNet loader.
        
        Args:
            model_path: Path to the NAFNet checkpoint file
            device: Device to load the model on
        """
        self.model_path = model_path
        self.device = device
        self.model: Optional[StandaloneNAFNet] = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the NAFNet model from checkpoint.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading standalone NAFNet from {self.model_path}")
            
            # Create model with NAFNet-Width64 configuration
            self.model = StandaloneNAFNet(
                img_channel=3,
                width=64,
                middle_blk_num=12,
                enc_blk_nums=[2, 2, 4, 8],
                dec_blk_nums=[2, 2, 2, 2]
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'params' in checkpoint:
                state_dict = checkpoint['params']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
            
            # Load state dict
            self.model.load_state_dict(new_state_dict, strict=True)
            
            # Move to device and set eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"Standalone NAFNet loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load standalone NAFNet: {e}")
            self.is_loaded = False
            return False
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process an image through the NAFNet model.
        
        Args:
            image: Input image as numpy array (H, W, 3) in BGR format
            
        Returns:
            Processed image as numpy array (H, W, 3) in BGR format
        """
        if not self.is_loaded or self.model is None:
            logger.warning("Model not loaded, returning original image")
            return image
        
        try:
            import cv2
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            image_normalized = image_rgb.astype(np.float32) / 255.0
            
            # Convert to tensor (C, H, W)
            image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)
            
            # Process through model
            with torch.no_grad():
                output_tensor = self.model(image_tensor)
            
            # Convert back to numpy
            output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Denormalize and convert to uint8
            output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
            
            # Convert RGB back to BGR
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
            
            return output_bgr
            
        except Exception as e:
            logger.error(f"Error processing image through standalone NAFNet: {e}")
            return image


def create_standalone_nafnet(model_path: str, device: str = "cuda") -> StandaloneNAFNetLoader:
    """
    Create and load a standalone NAFNet model.
    
    Args:
        model_path: Path to the NAFNet checkpoint
        device: Device to run on
        
    Returns:
        Loaded StandaloneNAFNetLoader instance
    """
    loader = StandaloneNAFNetLoader(model_path, device)
    loader.load_model()
    return loader


if __name__ == "__main__":
    # Test the standalone implementation
    logging.basicConfig(level=logging.INFO)
    
    print("🔬 Testing Standalone NAFNet Implementation")
    print("="*60)
    
    # Test model creation
    try:
        model = StandaloneNAFNet()
        print("✅ Standalone NAFNet model created successfully")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Forward pass successful: {dummy_input.shape} -> {output.shape}")
        
    except Exception as e:
        print(f"❌ Standalone NAFNet test failed: {e}")
    
    print("="*60)