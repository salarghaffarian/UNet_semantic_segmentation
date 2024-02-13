
"""
This module defines the U-Net model for image segmentation tasks.

The U-Net model implemented in this module can be used for both binary and multiclass segmentation. 
The number of output channels in the final layer of the model is determined by the `n_class` parameter in the `UNet` class.

If `n_class` is 1, the model is set up for binary segmentation. 
If `n_class` is greater than 1, the model is set up for multiclass segmentation. For example, if `n_class` is 2, the model will output two channels for the two classes in a binary segmentation task. 
If `n_class` is 3, the model will output three channels for a three-class segmentation task, and so on.

The `UNet` class in this module should be used to create an instance of the U-Net model with the desired number of output classes.
""" 

import torch
import torch.nn as nn

def double_conv(in_channels, out_channels, use_bn_dropout, dropout_p=0.5):
    layers = [
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    ]
    
    if use_bn_dropout:
        layers.insert(1, nn.BatchNorm2d(out_channels))
        layers.insert(3, nn.BatchNorm2d(out_channels))
        layers.insert(4, nn.Dropout(dropout_p))
        layers.insert(6, nn.Dropout(dropout_p))
    
    return nn.Sequential(*layers)


class UNet(nn.Module):
    """
    This class implements the U-Net architecture, which is a type of convolutional neural network (CNN) 
    used for tasks such as image segmentation. The architecture consists of a contracting path (encoder) 
    to capture context and a symmetric expanding path (decoder) that enables precise localization.

    Args:
        n_class (int): The number of output classes.
        image_channels (int, optional): The number of channels in the input images. Default is 3 (for RGB images).
        use_bn_dropout (bool, optional): Whether to use batch normalization and dropout layers in the model architecture. 
                                         Default is False (for having the original U-Net architecture).
        dropout_p (float, optional): The probability of an element to be zeroed if the dropout is used. Default is 0.5.

    Attributes:
        dconv_down1, dconv_down2, dconv_down3, dconv_down4 (nn.Sequential): Double convolutional layers in the encoder part.
        maxpool (nn.MaxPool2d): Max pooling layer.
        upsample (nn.Upsample): Upsampling layer.
        dconv_up3, dconv_up2, dconv_up1 (nn.Sequential): Double convolutional layers in the decoder part.
        conv_last (nn.Conv2d): The final convolutional layer.
    """

    def __init__(self, n_class, image_channels=3, use_bn_dropout=False, dropout_p=0.5):    # put use_bn_dropout=True and dropout_p=0.5 for better results which uses batch normalization and dropout layers in the model architecture. (put use_bn_dropout=False for having the original U-Net architecture)
        super().__init__()
                
        self.dconv_down1 = double_conv(image_channels, 64, use_bn_dropout, dropout_p)
        self.dconv_down2 = double_conv(64, 128, use_bn_dropout, dropout_p)
        self.dconv_down3 = double_conv(128, 256, use_bn_dropout, dropout_p)
        self.dconv_down4 = double_conv(256, 512, use_bn_dropout, dropout_p)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dconv_up3 = double_conv(256 + 512, 256, use_bn_dropout, dropout_p)
        self.dconv_up2 = double_conv(128 + 256, 128, use_bn_dropout, dropout_p)
        self.dconv_up1 = double_conv(128 + 64, 64, use_bn_dropout, dropout_p)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
    def forward(self, x):
        # x: [batch_size, image_channels, height, width]
        # For example: x: [batch_size, 3, 448, 448] >> assumes the input image size is 3x448x448.

        conv1 = self.dconv_down1(x)  # conv1: [batch_size, 64, height, width], conv1: [batch_size, 64, 448, 448]
        x = self.maxpool(conv1)      # x: [batch_size, 64, height/2, width/2], x: [batch_size, 64, 224, 224]

        conv2 = self.dconv_down2(x)  # conv2: [batch_size, 128, height/2, width/2], conv2: [batch_size, 128, 224, 224]
        x = self.maxpool(conv2)      # x: [batch_size, 128, height/4, width/4], x: [batch_size, 128, 112, 112]

        conv3 = self.dconv_down3(x)  # conv3: [batch_size, 256, height/4, width/4], conv3: [batch_size, 256, 112, 112]
        x = self.maxpool(conv3)      # x: [batch_size, 256, height/8, width/8], x: [batch_size, 256, 56, 56]

        x = self.dconv_down4(x)      # x: [batch_size, 512, height/8, width/8], x: [batch_size, 512, 56, 56]

        x = self.upsample(x)         # x: [batch_size, 512, height/4, width/4], x: [batch_size, 512, 112, 112]
        x = torch.cat([x, conv3], dim=1)  # x: [batch_size, 768, height/4, width/4], x: [batch_size, 768, 112, 112]

        x = self.dconv_up3(x)        # x: [batch_size, 256, height/4, width/4], x: [batch_size, 256, 112, 112]
        x = self.upsample(x)         # x: [batch_size, 256, height/2, width/2], x: [batch_size, 256, 224, 224]
        x = torch.cat([x, conv2], dim=1)  # x: [batch_size, 384, height/2, width/2], x: [batch_size, 384, 224, 224]

        x = self.dconv_up2(x)        # x: [batch_size, 128, height/2, width/2], x: [batch_size, 128, 224, 224]
        x = self.upsample(x)         # x: [batch_size, 128, height, width], x: [batch_size, 128, 448, 448]
        x = torch.cat([x, conv1], dim=1)  # x: [batch_size, 192, height, width], x: [batch_size, 192, 448, 448]

        x = self.dconv_up1(x)        # x: [batch_size, 64, height, width], x: [batch_size, 64, 448, 448]

        out = self.conv_last(x)      # out: [batch_size, n_class, height, width], out: [batch_size, n_class, 448, 448]

        return out
    



