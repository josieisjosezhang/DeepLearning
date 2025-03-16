import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer import TransformerEncoder

def make_diviserable(v,divisor,min_value=None):
    ''''
    确保所有通道都是可以被divisor整除的
    '''
    if min_value is None:
        min_value =divisor
    new_v = max(min_value,int(v+divisor/2)//divisor*divisor)
    if new_v<0.9*v:
        new_v += divisor
    return new_v
class ConvLayer(nn.Module):
    ''''
    网络中的卷积结构
    '''
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,groups=1,bias=False,
                 use_norm=True,use_act=True):
        super(ConvLayer, self).__init__()

        if isinstance(kernel_size,int):
            kernel_size=(kernel_size,kernel_size)
        if isinstance(stride,int):
            stride=(stride,stride)
        assert isinstance(kernel_size,tuple)
        assert isinstance(stride,tuple)

        padding=(int((kernel_size[0]-1)/2),int((kernel_size[1]-1)/2))

        block=nn.Sequential()

        conv_layer=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,
                             groups=groups,bias=bias)
        block.add_module(name='conv',module=conv_layer)

        if use_norm:
            norm_layer=nn.BatchNorm2d(out_channels,momentum=0.1)
            block.add_module(name='norm',module=norm_layer)

        if use_act:
            act_layer=nn.SiLU()
            block.add_module(name='act',module=act_layer)

        self.block=block

    def forward(self, x):
        return self.block(x)

class InvertedResidual(nn.Module):
    def __init__(self,in_channels,out_channels,stride,expand_ratio,skip_connection=True):
        super(InvertedResidual, self).__init__()
        assert stride==1 or stride==2
        hidden_dim=make_diviserable(int(round(in_channels*expand_ratio)),8)#表示通过MV2的第一个1*1卷积后通道被变成了多少

        block=nn.Sequential()

        if expand_ratio!=1:
            block.add_module(name='expand_conv_1*1',module=ConvLayer(in_channels,hidden_dim,kernel_size=1))

        block.add_module(name='conv_3*3',module=ConvLayer(hidden_dim,hidden_dim,kernel_size=3,stride=stride,groups=hidden_dim))
        #由于采用的是深度可分离卷积，所以groups=hidden_dim

        block.add_module(name='conv_1*1',module=ConvLayer(hidden_dim,out_channels,kernel_size=1,use_act=False,use_norm=True))

        self.blocks=block
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.exp=expand_ratio
        self.stride=stride
        self.use_res_connection=(self.stride==1 and in_channels==out_channels and skip_connection)#是否需要使用捷径分支

    def forward(self,x):
        if self.use_res_connection:
            return x+self.block(x)
        else:
            return self.block(x)

class MobileViT(nn.Module):
    def __init__(self,in_channels,transformer_dim,ffn_dim,num_transformer_block,head_dim,attn_drop_ratio=0.0,dropout=0.0,
                 ffn_drop_ratio=0.0,patch_h=8,patch_w=8,conv_ksize=3):
        super(MobileViT, self).__init__()
        '''
        transformer_dim: 每个token对应的序列长度
        ffn_dim：MLP中第一个全连接节点的隐藏层维度
        '''
        #local_representation
        conv_3x3_in=ConvLayer(in_channels,in_channels,3,1)
        conv_1x1_in=ConvLayer(in_channels,transformer_dim,1,1,use_norm=False,use_act=False)

        self.local_rep=nn.Sequential()
        self.local_rep.add_module(name='conv_3x3_in',module=conv_3x3_in)
        self.local_rep.add_module(name='conv_1x1_in',module=conv_1x1_in)

        assert transformer_dim%head_dim==0
        num_heads=transformer_dim//head_dim

        global_rep=[
            TransformerEncoder(embed_dim=transformer_dim,
                               ffn_latent_dim=ffn_dim,
                               num_heads=num_heads,
                               attn_drop_ratio=attn_drop_ratio,
                               dropout=dropout,
                               ffn_drop_ratio=ffn_drop_ratio)
            for _ in range(num_transformer_block)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep=nn.Sequential(*global_rep)

        #fusion
        conv_1x1_out=ConvLayer(transformer_dim,in_channels,1,1)
        conv_3x3_out=ConvLayer(2*in_channels,in_channels,conv_ksize,1)

        self.conv_proj=conv_1x1_out
        self.fusion=conv_3x3_out

        self.patch_h=patch_h
        self.patch_w=patch_w
        self.patch_area=self.patch_h*self.patch_w

        self.cnn_in_dim=in_channels
        self.cnn_out_dim=transformer_dim
        self.head=num_heads
        self.ffn_dim=ffn_dim
        self.dropout=dropout
        self.ffn_drop_ratio=ffn_drop_ratio
        self.attn_drop_ratio=attn_drop_ratio
        self.n_blocks=num_transformer_block
        self.conv_ksize=conv_ksize

    def unfolding(self,x):
        patch_h=self.patch_h
        patch_w=self.patch_w
        patch_area=self.patch_h*self.patch_w

        bs,c,original_h,original_w=x.shape

        new_h=int(math.ceil(original_h/self.patch_h)*self.patch_h)
        new_w=int(math.ceil(original_w/self.patch_w)*self.patch_w)

        interpolate=False
        if new_w!=original_w or new_h!=original_h:
            #使用双线性插值
            x=F.interpolate(x,size=(new_w,new_h),mode='bilinear',align_corners=False)
            interpolate=True

        num_patch_h=new_h//self.patch_h
        num_patch_w=new_w//self.patch_w
        num_patches=num_patch_h*num_patch_w

        #进行unfold，token数量=num_patches
        #->[batchsize,channel,num_patch_h,patch_size_h,num_patch_w,patch_size_w]
        x=x.reshape(bs,c,num_patch_h,patch_h,num_patch_w,patch_w)
        #->[batchsize,channel,num_patch_h,num_patch_w,patch_size_h,patch_size_w]
        x=x.transpose(3,4)
        #->[batchsize,channel,num_patch,patch_area]
        x=x.reshape(bs,c,num_patches,patch_area)
        x=x.transpose(1,3)
        #->[batchsize*patch_area,num_patch,channel]
        x=x.reshape(bs*patch_area,num_patches,-1)

        #构造原始高宽信息记录
        info_dict={
            "original_size": (original_h,original_w),
            "batch_size": bs,
            "interpolate": interpolate,
            "num_patches": num_patches,
            "num_patch_h": num_patch_h,
            "num_patch_w": num_patch_w
        }

        return x,info_dict

    def folding(self,x,info_dict):
        n_dim=x.dim()#确定输入的维度
        assert n_dim==3

        #->[batchsize,patch_area,num_patches,c]
        x=x.contiguous().view(
            info_dict['batch_size'],self.patch_area,info_dict['num_patches'],-1
        )
        bs,pixal,num_patches,channel=x.shape
        num_patch_h=info_dict['num_patch_h']
        num_patch_w=info_dict['num_patch_w']
        #->[batchsize,channel,num_patches,pixal]
        x=x.transpose(1,3)
        x=x.reshape(bs*channel*num_patch_h,num_patch_w,self.patch_h,self.patch_w)
        #->[batchsize*channel*num_patches_h,patch_h,num_patch_w,patch_w]
        x=x.transpose(1,2)
        if info_dict['interpolate']:
            x=F.interpolate(x,size=(info_dict['original_size']),mode='bilinear',align_corners=False)
        return x

    def forward(self,x):
        res=x

        fm=self.local_rep(x)

        patches,info_dict=self.unfolding(x)
        #通过transformer
        for i in self.global_rep:
            patches=i(patches)

        fm=self.folding(fm,info_dict)
        fm=self.conv_proj(fm)
        fm=self.fusion(torch.concat(fm,res),dim=1)
        return fm


class MobileViT(nn.Module):
    def __init__(self,model_config,num_classes):
        super(MobileViT, self).__init__()

        img_channels=3
        out_channels=16 #只是初始化，第一个卷积层的输出是16

        self.conv1=ConvLayer(in_channels=img_channels,out_channels=out_channels,kernel_size=3,stride=2)

        self.layer1,out_channels=self._make_layer(input_channels=out_channels,cfg=model_config['layer1'])
        self.layer2,out_channels=self._make_layer(input_channels=out_channels,cfg=model_config['layer2'])
        self.layer3,out_channels=self._make_layer(input_channels=out_channels,cfg=model_config['layer3'])
        self.layer4,out_channels=self._make_layer(input_channels=out_channels,cfg=model_config['layer4'])
        self.layer5,out_channels=self._make_layer(input_channels=out_channels,cfg=model_config['layer5'])

        exp_channels=min(model_config['last_layer_exp_factor']*out_channels,960)
        self.conv_1x1_exp=ConvLayer(in_channels=out_channels,out_channels=exp_channels,kernel_size=1)
        self.classifier=nn.Sequential()
        self.classifier.add_module('global_pool',nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module('flatten',nn.Flatten())

        if 0.0<model_config['cls_dropout']<1.0:
            self.classifier.add_module('dropout',nn.Dropout(model_config['cls_dropout']))
        self.classifier.add_module('fc',nn.Linear(exp_channels,num_classes))

        self.apply(self.init_parameters)

    def make_layer(self,input_channels,cfg):
        block_type=cfg.get('block_type','mobilevit')
        if block_type.lower() == 'mobilevit':
            return self._make_mit_layer(input_channels,cfg)
        else:
            return self._make_layer(input_channels,cfg)

    def _make_mobilenet_layer(self,input_channels,cfg):
        output_channels=cfg.get('output_channels')
        num_blocks=cfg.get('num_blocks',2)
        expand_ratio=cfg.get('expand_ratio',4)
        block=[]
        for i in range(num_blocks):
            stride=cfg.get('stride',1) if i==0 else 1
            layer=InvertedResidual(input_channels,output_channels,stride,expand_ratio)
            block.append(layer)
            input_channels=output_channels

        return nn.Sequential(*block)

    def _make_mit_layer(self,input_channels,cfg):
        stride=cfg.get('stride',1)
        block=[]
        if stride==2:
            layer=InvertedResidual(in_channels=input_channels,
                                   out_channels=cfg.get('output_channels'),
                                   stride=stride,
                                   expand_ratio=cfg.get('expand_ratio',4))
            block.append(layer)
            input_channels=cfg.get('output_channels')
        transformer_dim=cfg["transformer_channels"]
        ffn_dim=cfg.get('ffn_dim')
        num_heads=cfg.get('num_heads',4)
        head_dim=transformer_dim//num_heads

        if transformer_dim%num_heads!=0:
            raise ValueError('transformer input dimension should be divisible by head numbers')

        block.extend([MobileViT(input_channels,transformer_dim,ffn_dim,
                                cfg.get('transformer_blocks',1),attn_drop_ratio=cfg.get(),dropout=cfg.get('dropout',0.1),
                                ffn_drop_ratio=cfg.get('ffn_dropout',0.0),patch_h=cfg.get('patch_h',2),
                                patch_w=cfg.get('patch_w',2),head_dim=head_dim,conv_ksize=3)
                      ])
        return nn.Sequential(*block),input_channels

    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self,x):
        x=self.conv1(x)
        x=self.lay1(x)
        x=self.lay2(x)

        x=self.lay3(x)
        x=self.lay4(x)
        x=self.lay5(x)
        x=self.conv_1x1_exp(x)
        x=self.classifier(x)
        return x
