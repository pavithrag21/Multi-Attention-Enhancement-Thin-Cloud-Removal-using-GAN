import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from models.models_utils import weights_init, print_network
from global_local_attention_module_pytorch.local_channel_attention import LocalChannelAttention
from global_local_attention_module_pytorch.global_channel_attention import GlobalChannelAttention
from global_local_attention_module_pytorch.local_spatial_attention import LocalSpatialAttention
from global_local_attention_module_pytorch.global_spatial_attention import GlobalSpatialAttention
from src.attention_util import AttentionGate, ChannelGate, Flatten
# import common

###### Layer 
def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 1,
                    stride =stride, padding=0,bias=False)

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
        stride =stride, padding=1,bias=False)

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,):
        super(Bottleneck,self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False,dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x) 
        return out

class irnn_layer(nn.Module):
    def __init__(self,in_channels):
        super(irnn_layer,self).__init__()
        self.left_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.right_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.up_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.down_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        
    def forward(self,x):
        _,_,H,W = x.shape
        top_left = x.clone()
        top_right = x.clone()
        top_up = x.clone()
        top_down = x.clone()
        top_left[:,:,:,1:] = F.relu(self.left_weight(x)[:,:,:,:W-1]+x[:,:,:,1:],inplace=False)
        top_right[:,:,:,:-1] = F.relu(self.right_weight(x)[:,:,:,1:]+x[:,:,:,:W-1],inplace=False)
        top_up[:,:,1:,:] = F.relu(self.up_weight(x)[:,:,:H-1,:]+x[:,:,1:,:],inplace=False)
        top_down[:,:,:-1,:] = F.relu(self.down_weight(x)[:,:,1:,:]+x[:,:,:H-1,:],inplace=False)
        return (top_up,top_right,top_down,top_left)


class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention,self).__init__()
        self.out_channels = int(in_channels/2)
        self.conv1 = nn.Conv2d(in_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels,4,kernel_size=1,padding=0,stride=1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


class SAM(nn.Module):
    def __init__(self,in_channels,out_channels,attention=1):
        super(SAM,self).__init__()
        self.out_channels = out_channels
        self.irnn1 = irnn_layer(self.out_channels)
        self.irnn2 = irnn_layer(self.out_channels)
        self.conv_in = conv3x3(in_channels,self.out_channels)
        self.relu1 = nn.ReLU(True)
        
        self.conv1 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(self.out_channels*4,self.out_channels,kernel_size=1,stride=1,padding=0)
        self.conv3 = nn.Conv2d(self.out_channels*4,self.out_channels,kernel_size=1,stride=1,padding=0)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels,1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv1(x)
        top_up,top_right,top_down,top_left = self.irnn1(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:,0:1,:,:])
            top_right.mul(weight[:,1:2,:,:])
            top_down.mul(weight[:,2:3,:,:])
            top_left.mul(weight[:,3:4,:,:])
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv2(out)
        top_up,top_right,top_down,top_left = self.irnn2(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:,0:1,:,:])
            top_right.mul(weight[:,1:2,:,:])
            top_down.mul(weight[:,2:3,:,:])
            top_left.mul(weight[:,3:4,:,:])
        
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        mask = self.sigmod(self.conv_out(out))
        return mask
    
# class SimpleCNN(nn.Module):
#     def __init__(self, in_channels):
#         super(SimpleCNN, self).__init__()
#         out_channels = 32
#         # Define a single Conv2d layer
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
#     def forward(self, x):   # Apply Conv2d layer followed by ReLU activation
#         x = F.relu(self.conv1(x))
#         return x

###### Network
class SPANet(nn.Module):
    def __init__(self):
        super(SPANet,self).__init__()

        self.conv_in = nn.Sequential(
            conv3x3(3,32),
            nn.ReLU(True)
            )
                
        self.SAM1 = SAM(32,32,1)
        self.res_block1 = Bottleneck(32,32)
        self.res_block2 = Bottleneck(32,32)
        self.res_block3 = Bottleneck(32,32)
        self.res_block4 = Bottleneck(32,32)
        self.res_block5 = Bottleneck(32,32)
        self.res_block6 = Bottleneck(32,32)
        self.res_block7 = Bottleneck(32,32)
        self.res_block8 = Bottleneck(32,32)
        self.res_block9 = Bottleneck(32,32)
        self.res_block10 = Bottleneck(32,32)
        self.res_block11 = Bottleneck(32,32)
        self.res_block12 = Bottleneck(32,32)
        self.res_block13 = Bottleneck(32,32)
        self.res_block14 = Bottleneck(32,32)
        self.res_block15 = Bottleneck(32,32)
        self.res_block16 = Bottleneck(32,32)
        self.res_block17 = Bottleneck(32,32)
        self.conv_out = nn.Sequential(
            conv3x3(32,3)
        )
        self.cbam = CBAM(gate_channels = 32, kernel_size = 3, reduction_ratio = 16, pool_types = ['avg', 'max'], no_spatial = False, bam = False, num_layers = 1, bn = False, dilation_conv_num = 2, dilation_val = 4)
        
        self.glam = GLAM(in_channels=32, num_reduced_channels=16, feature_map_size=64, kernel_size=5)
        self.avg=nn.AvgPool2d(3, stride=8)
        self.up= nn.Upsample(scale_factor=8, mode='bilinear')
        self.conv2d1 = nn.Conv2d(64,32,kernel_size=1,stride=1,padding=0)
        self.conv2d2 = nn.Conv2d(64,32,kernel_size=1,stride=1,padding=0)
        self.conv2d3 = nn.Conv2d(64,32,kernel_size=1,stride=1,padding=0)
        self.relu11 = nn.ReLU()


    def forward(self, x):

        out = self.conv_in(x)
        out = F.relu(self.res_block1(out) + out)
        out = F.relu(self.res_block2(out) + out)
        out = F.relu(self.res_block3(out) + out)
            
        #for iteration in range(3):
        out_cbam1=self.cbam(out)#output for cbam
        out_down1 = self.avg(out)
        out_glam1=self.glam(out_down1) #torch.Size([1, 32, 512, 512])(batch_size,in_channels,height,width)
        out_up1=self.up(out_glam1)#output for glam
        concatenated_tensor1 = torch.cat((out_cbam1, out_up1), dim=1)  # Concatenate along the channel dim
#         print(concatenated_tensor.shape) #torch.Size([1, 64, 512, 512])
        out_conv2d1 = self.conv2d1(concatenated_tensor1)
#         print(out_conv2d1.shape)
#         print("executed 1")
        
        
        out_cbam2=self.cbam(out_conv2d1)#output for cbam
        out_down2 = self.avg(out_conv2d1)
        out_glam2=self.glam(out_down2) #torch.Size([1, 32, 512, 512])(batch_size,in_channels,height,width)
        out_up2=self.up(out_glam2)#output for glam
        concatenated_tensor2 = torch.cat((out_cbam2, out_up2), dim=1)  # Concatenate along the channel dim
#         print(concatenated_tensor.shape) #torch.Size([1, 64, 512, 512])
        out_conv2d2 = self.conv2d2(concatenated_tensor2)
#         print(out_conv2d2.shape)
#         print("executed 2")
        
        out_cbam3=self.cbam(out_conv2d2)#output for cbam
        out_down3 = self.avg(out_conv2d2)
        out_glam3=self.glam(out_down3) #torch.Size([1, 32, 512, 512])(batch_size,in_channels,height,width)
        out_up3=self.up(out_glam3)#output for glam
        concatenated_tensor3 = torch.cat((out_cbam3, out_up3), dim=1)  # Concatenate along the channel dim
#         print(concatenated_tensor.shape) #torch.Size([1, 64, 512, 512])
        out_conv2d3 = self.conv2d3(concatenated_tensor3)
#         print(out_conv2d3.shape)
#         print("executed 3")
       
        #print(iteration)
        # _, in_channels, _, _ = concatenated_tensor
#         input_tensor = torch.randn(concatenated_tensor.shape)
#         in_channels = concatenated_tensor.shape[1]

        # Create the network with the specified input channels and fixed output channels (32)
#         model = SimpleCNN(in_channels=in_channels)
#         out= model(input_tensor)
#         print(out.shape)
#         exit()
        out = out_conv2d3 + out
#         print(out.shape)
        
        
        Attention1 = self.SAM1(out) 
        out = F.relu(self.res_block4(out) * Attention1  + out)
        out = F.relu(self.res_block5(out) * Attention1  + out)
        out = F.relu(self.res_block6(out) * Attention1  + out)
        
        Attention2 = self.SAM1(out) 
        out = F.relu(self.res_block7(out) * Attention2 + out)
        out = F.relu(self.res_block8(out) * Attention2 + out)
        out = F.relu(self.res_block9(out) * Attention2 + out)
        
        Attention3 = self.SAM1(out) 
        out = F.relu(self.res_block10(out) * Attention3 + out)
        out = F.relu(self.res_block11(out) * Attention3 + out)
        out = F.relu(self.res_block12(out) * Attention3 + out)
        
        Attention4 = self.SAM1(out) 
        out = F.relu(self.res_block13(out) * Attention4 + out)
        out = F.relu(self.res_block14(out) * Attention4 + out)
        out = F.relu(self.res_block15(out) * Attention4 + out)
        
        out = F.relu(self.res_block16(out) + out)
        out = F.relu(self.res_block17(out) + out)
       
        out = self.conv_out(out)

        return Attention4 , out

class Generator(nn.Module):
    def __init__(self, gpu_ids):
        super().__init__()
        self.gpu_ids = gpu_ids

        self.gen = nn.Sequential(OrderedDict([('gen', SPANet())]))
        self.gen.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids:
            return nn.parallel.data_parallel(self.gen, x, self.gpu_ids)
        else:
            return self.gen(x)



### GLAM Module starts here


class GLAM(nn.Module):
    
    def __init__(self, in_channels, num_reduced_channels, feature_map_size, kernel_size):
        '''
        Song, C. H., Han, H. J., & Avrithis, Y. (2022). All the attention you need: Global-local, spatial-channel attention for image retrieval. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 2754-2763).

        Args:
            in_channels (int): number of channels of the input feature map
            num_reduced_channels (int): number of channels that the local and global spatial attention modules will reduce the input feature map. Refer to figures 3 and 5 in the paper.
            feaure_map_size (int): height/width of the feature map
            kernel_size (int): scope of the inter-channel attention
        '''
        
        super().__init__()
        
        self.local_channel_att = LocalChannelAttention(feature_map_size, kernel_size)
        self.local_spatial_att = LocalSpatialAttention(in_channels, num_reduced_channels)
        self.global_channel_att = GlobalChannelAttention(feature_map_size, kernel_size)
        self.global_spatial_att = GlobalSpatialAttention(in_channels, num_reduced_channels)
        
        self.fusion_weights = nn.Parameter(torch.Tensor([0.333, 0.333, 0.333])) # equal intial weights
        
    def forward(self, x):
        local_channel_att = self.local_channel_att(x) # local channel
        local_att = self.local_spatial_att(x, local_channel_att) # local spatial
        global_channel_att = self.global_channel_att(x) # global channel
        global_att = self.global_spatial_att(x, global_channel_att) # global spatial
        
        local_att = local_att.unsqueeze(1) # unsqueeze to prepare for concat
        global_att = global_att.unsqueeze(1) # unsqueeze to prepare for concat
        x = x.unsqueeze(1) # unsqueeze to prepare for concat
        
        all_feature_maps = torch.cat((local_att, x, global_att), dim=1)
        weights = self.fusion_weights.softmax(-1).reshape(1, 3, 1, 1, 1)
        fused_feature_maps = (all_feature_maps * weights).sum(1)
        
        return fused_feature_maps

### GLAM ends starts here

### CBAM Module starts here

class SpatialGate(nn.Module):
    def __init__(
        self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4
    ):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module(
            "gate_s_conv_reduce0",
            nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1),
        )
        self.gate_s.add_module(
            "gate_s_bn_reduce0", nn.BatchNorm2d(gate_channel // reduction_ratio)
        )
        self.gate_s.add_module("gate_s_relu_reduce0", nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                "gate_s_conv_di_%d" % i,
                nn.Conv2d(
                    gate_channel // reduction_ratio,
                    gate_channel // reduction_ratio,
                    kernel_size=3,
                    padding=dilation_val,
                    dilation=dilation_val,
                ),
            )
            self.gate_s.add_module(
                "gate_s_bn_di_%d" % i, nn.BatchNorm2d(gate_channel // reduction_ratio)
            )
            self.gate_s.add_module("gate_s_relu_di_%d" % i, nn.ReLU())
        self.gate_s.add_module(
            "gate_s_conv_final",
            nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1),
        )

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)



class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels,
        kernel_size=3,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
        bam=False,
        num_layers=1,
        bn=False,
        dilation_conv_num=2,
        dilation_val=4,
    ):
        super(CBAM, self).__init__()
        self.bam = bam
        self.no_spatial = no_spatial
        if self.bam:
            self.dilatedGate = SpatialGate(
                gate_channels, reduction_ratio, dilation_conv_num, dilation_val
            )
            self.ChannelGate = ChannelGate(
                gate_channels,
                reduction_ratio,
                pool_types,
                bam=self.bam,
                num_layers=num_layers,
                bn=bn,
            )
        else:
            self.ChannelGate = ChannelGate(
                gate_channels, reduction_ratio, pool_types
            )
            if not no_spatial:
                self.SpatialGate = AttentionGate(kernel_size)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.bam:
            if not self.no_spatial:
                x_out = self.SpatialGate(x_out)
            return x_out
        else:
            att = 1 + F.sigmoid(self.ChannelGate(x) * self.dilatedGate(x))
            return att * x
#CBAM Ends here


