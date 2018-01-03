import torch
import torch.nn as nn

"""
    < class ConvBlock >
    
    It consists of Convolution - Norm - Activation
"""
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=1, pad=0, bn=True, act_type='relu'):
        super(ConvBlock, self).__init__()
        
        layer_list = []
        layer_list += [nn.Conv2d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=pad)]
        
        # Make BatchNorm layer
        if bn == True:
            layer_list += [nn.BatchNorm2d(out_dim, affine=True)]
        
        # Make activation layer
        if act_type == 'relu':
            layer_list += [nn.ReLU()]
        elif act_type == 'leakyrelu':
            layer_list += [nn.LeakyReLU(negative_slope=0.01)]
        elif act_type == 'tanh':
            layer_list += [nn.Tanh()]
        elif act_type == None:
            pass
        
        self.conv_block = nn.Sequential(*layer_list)
        
    def forward(self, x):
        out = self.conv_block(x)
        return out

    
"""
    < class ResBlock >

    It consists of two ConvBlocks and uses Identity Mapping(resnet).
"""
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=1, pad=1):            
        super(ResBlock, self).__init__()
        
        conv_block_1 = ConvBlock(in_dim, out_dim, kernel=kernel, stride=stride, pad=pad, 
                                 bn=True, act_type='relu')
        conv_block_2 = ConvBlock(in_dim, out_dim, kernel=kernel, stride=stride, pad=pad,
                                 bn=True, act_type=None)
        
        self.res_block = nn.Sequential(conv_block_1, conv_block_2)
    
    def forward(self, x):
        out = x + self.res_block(x)
        return out
        
        
"""
    < class ConvTransBlock >

    It consists of Transpose Convolution - Batch Norm - ReLU
"""
class ConvTransBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=2, pad=1, output_pad=1):
        super(ConvTransBlock, self).__init__()

        conv_trans = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel, stride=stride, 
                                        padding=pad, output_padding=output_pad)
        norm = nn.BatchNorm2d(out_dim, affine=True)
        relu = nn.ReLU()
        
        self.deconv_block = nn.Sequential(conv_trans, norm, relu)
            
    def forward(self, x):
        out = self.deconv_block(x)
        return out
    
    
"""
    <class Discriminator>
    
    After five ConvBlocks, it splits into two parts.
    
    Part 1. PatchGAN
        In this part, size of the last activation volue is (N, 512, 4, 4)
        The discriminator should give scores(fake : 0, real : 1) to all of these 16(4 x 4) patches.
        
    Part 2. Gender Classifier
        It consists of a few of linear blocks classifying between male and female.
        
    These parts will be added at the lst to make a loss and trained with multitask learning.
"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # (N, 3, 128, 128) -> (N, 32, 64, 64)
        conv_1 = ConvBlock(3, 32, kernel=4, stride=2, pad=1, bn=False, act_type='leakyrelu') 
        # (N, 32, 64, 64) -> (N, 64, 32, 32)
        conv_2 = ConvBlock(32, 64, kernel=4, stride=2, pad=1, bn=True, act_type='leakyrelu')
        # (N, 64, 32, 32) -> (N, 128, 16, 16)
        conv_3 = ConvBlock(64, 128, kernel=4, stride=2, pad=1, bn=True, act_type='leakyrelu')
        # (N, 128, 16, 16) -> (N, 256, 8, 8)
        conv_4 = ConvBlock(128, 256, kernel=4, stride=2, pad=1, bn=True, act_type='leakyrelu')
        # (N, 256, 8, 8) -> (N, 512, 4, 4)
        conv_5 = ConvBlock(256, 512, kernel=4, stride=2, pad=1, bn=True, act_type='leakyrelu')     
        
        self.conv_blocks = nn.Sequential(conv_1, conv_2, conv_3, conv_4, conv_5)
        
        # Part 1. PatchGAN
        # (N, 512, 4, 4) -> (N, 1, 4, 4)
        self.patch_gan = ConvBlock(512, 1, kernel=3, stride=1, pad=1, bn=False, act_type=None)
                
        # Part 2. Gender Classifier
        # (N, 512, 4, 4) -> (N, 1000)
        fc_1 = nn.Sequential(nn.Linear(512 * 4 * 4, 1000), nn.ReLU())        
        # (N, 1000) -> (N, 100)
        fc_2 = nn.Sequential(nn.Linear(1000, 100), nn.ReLU())
        # (N, 100) -> (N, 2)
        fc_3 = nn.Linear(100, 2)
        
        self.gender_classifier = nn.Sequential(fc_1, fc_2, fc_3)

    def forward(self, x):
        out = self.conv_blocks(x)
        gan_score = self.patch_gan(out)
        cls_score = self.gender_classifier(out.view(-1, 512 * 4 * 4))
        
        return gan_score, cls_score

"""
    < class Generator >
    
    Part 1. Downsalmple
        3 ConvBlocks
        
    Part 2. Res Blocks
        2 ResBlocks 
        
    Part 3. Upsample
        2 ConvTransBlocks - 1 ConvBlock
"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Part 1. Downsample
        # (N, 3, 128, 128) -> (N, 32, 128, 128)
        conv_1 = ConvBlock(3, 32, kernel=7, stride=1, pad=3, bn=True, act_type='relu')    
        # (N, 32, 128, 128) -> (N, 64, 64, 64)
        conv_2 = ConvBlock(32, 64, kernel=3, stride=2, pad=1, bn=True, act_type='relu') 
        # (N, 64, 64, 64) -> (N, 128, 32, 32)
        conv_3 = ConvBlock(64, 128, kernel=3, stride=2, pad=1, bn=True, act_type='relu')
        
        self.downsample_blocks = nn.Sequential(conv_1, conv_2, conv_3)
        
        # Part 2. Res Blocks
        # (N, 128, 32, 32) -> (N, 128, 32, 32)
        res_1 = ResBlock(128, 128, kernel=3, stride=1, pad=1)
        # (N, 128, 32, 32) -> (N, 128, 32, 32)
        res_2 = ResBlock(128, 128, kernel=3, stride=1, pad=1)
        
        self.res_blocks = nn.Sequential(res_1, res_2)
        
        # Part 3. Upsample
        # (N, 128, 32, 32) -> (N, 64, 64, 64)
        conv_trans_1 = ConvTransBlock(128, 64, kernel=3, stride=2, pad=1, output_pad=1)
        # (N, 64, 64, 64) -> (N, 32, 128, 128)
        conv_trans_2 = ConvTransBlock(64, 32, kernel=3, stride=2, pad=1, output_pad=1)       
        # (N, 32, 128, 128) -> (N, 3, 128, 128)
        conv_4 = ConvBlock(32, 3, kernel=7, stride=1, pad=3, bn=False, act_type='tanh')
        
        self.upsample_blocks = nn.Sequential(conv_trans_1, conv_trans_2, conv_4)
    
    def forward(self, x):
        out = self.downsample_blocks(x)
        out = self.res_blocks(out)
        out = self.upsample_blocks(out)

        return out