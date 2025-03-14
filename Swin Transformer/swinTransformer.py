import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import SwinTransformerBlock
import torch.utils.checkpoint as checkpoint
import numpy as np

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def window_partition(x, window_size):
    ''''
    将feature_map按照window_size划分成一个个没有重叠的window
    x: b,h,w,c
    window_size: int
    '''
    b,h,w,c=x.shape
    x=x.view(b,h//window_size,window_size,w//window_size,window_size,c)
    windows=x.permute(0,1,3,2,5).contiguous().view(-1,window_size,window_size,c)#[b*num_windows,h,w,c]
    return windows

def window_reverse(windows,window_size,h,w):
    ''''
    windows: [b*num_windows,h,w,c]
    window_size: int
    h: img size
    w: image size
    '''
    b=int(windows.shape[0]/(h*w/window_size/window_size))
    x=windows.view(b,h//window_size,w//window_size,window_size,window_size,-1)
    x=x.permute(0,1,3,2,4,5).contiguous().view(b,h,w,-1)
    return  x

class PatchEmbedding(nn.Module):
    ''''
    patch partition+linear embedding
    '''
    def __init__(self,patch_size,in_c,embed_dim,norm_layer):
        super(PatchEmbedding, self).__init__()
        patch_size=(patch_size,patch_size)
        self.patch_size=patch_size
        self.in_c=in_c
        self.embed_dim=embed_dim
        self.proj=nn.Conv2d(in_c,embed_dim,kernel_size=patch_size,stride=patch_size)
        self.norm=norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self,x):
        _,_,h,w=x.shape
        #如果输入图像不是patch_size的整数倍就要进行padding
        pad_input=(h% self.patch_size[0]!=0) or (w% self.patch_size[1] !=0)
        if pad_input:
            #在图像的宽度右侧和高度下方进行pad的
            #(w_left,w_right,h_top,h_bottom,c_front,c_back)
            x=F.pad(x,(0,self.patch_size[1]-w%self.patch_size[1],0,self.patch_size[0]-w%self.patch_size[0],0,0))
        #利用卷积层 下采样 或者说是分割patch
        x=self.proj(x)
        _,_,h,w=x.shape
        #flatten: [b,c,h,w]->[b,c,hw]  这里的c已经是patch的维度了，hw是patch的个数
        #transpose: [b,c,hw]->[b,hw,c]
        x=x.flatten(2).transpose(1,2)
        x=self.norm(x)
        return x,h,w
class PatchMerging(nn.Module):
    ''''
    用来做downsample的，把图片宽高减小一半，通道数增加一倍
    包含concat\ln\linear的组件
    '''
    def __init__(self,dim,norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.dim=dim
        self.norm = norm_layer(4 * dim)
        self.reduction=nn.Linear(4*dim,2*dim)

    def forward(self,x,h,w):
        #x:[b,h*w,c]
        b,l,c=x.shape
        assert l==h*w,'input feature has wrong size'
        x=x.view(b,h,w,c)

        #padding，因为在高和宽下采样2倍
        pad_input=(h%2==1) or(w%2==1)
        if pad_input:
            #这里pad要对最后3个维度进行,传入的维度是从后往前传的
            #[c_front,c_back,w_front,w_back,h_front,h_back]
            x=F.pad(x,(0,0,0,w%2,0,h%2))

        x0=x[:,0::2,0::2,:]#batch取所有，hw都从0开始选，以2为间隔
        x1=x[:,1::2,0::2,:]
        x2=x[:,0::2,1::2,:]
        x3=x[:,1::2,1::2,:]
        x=torch.cat([x0,x1,x2,x3],dim=-1)#在最后一个维度 也就是channel维度拼接
        x=x.view(b,-1,4*c)

        x=self.norm(x)
        x=self.reduction(x)
        return

class MLP(nn.Module):
    def __init__(self,in_features,hidden_features,act_layer=nn.GELU,drop=0.):
        super(MLP,self).__init__()
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act1=act_layer()
        self.drop1=nn.Dropout(drop)
        self.fc2=nn.Linear(hidden_features,in_features)
        self.drop2=nn.Dropout(drop)
    def forward(self,x):
        x=self.fc1(x)
        x=self.act1(x)
        x=self.drop1(x)
        x=self.fc2(x)
        x=self.drop2(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self,dim,window_size,num_heads,qkv_bias=True,attn_drop=0.,proj_drop=0.):
        super(WindowAttention,self).__init__()
        self.dim=dim
        self.window_size=window_size
        self.num_heads=num_heads
        head_dim = dim // num_heads
        self.scale=head_dim ** -0.5

        self.relative_position_bias_table=nn.Parameter(torch.zeros((2*window_size[0]-1)*(2*window_size[1]-1),num_heads))

        coords_h=torch.arange(self.window_size[0])
        coords_w=torch.arange(self.window_size[1])
        coords=torch.stack(torch.meshgrid([coords_w,coords_h],indexing='ij'))#创建网格 [2,h,w]
        coords_flatten=coords.view(coords,1)#[2,hw]

        relative_coords=coords_flatten[:,:,None]-coords_flatten[:,None,:]#[2,hw,hw]
        relative_coords=relative_coords.permute(1,2,0).contiguous()
        relative_coords[:,:,0]+=self.window_size[0]-1
        relative_coords[:,:,1]+=self.window_size[1]-1
        relative_coords[:,:,0]*=2*self.window_size[1]-1
        relative_position_index=relative_coords.sum(-1)
        self.register_buffer('relative_position_index',relative_position_index)

        self.qkv=nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table,std=0.02)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,x,mask=None):
        b_,n,c=x.shape
        qkv=self.qkv(x).reshape(b_,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)

        q=q*self.scale
        attn=(q@k.transpose(-2,-1))

        relative_position_bias=self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0]*self.window_size[1],
                                                                                                              self.window_size[0]*self.window_size[1],-1)
        relative_position_bias=relative_position_bias.premute(2,0,1).contiguous()
        attn=attn+relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW=mask.shape[0]
            attn=attn.view(b_//nW,nW,self.num_heads,n,n)+mask.unsqueeze(1).unsqueeze(0)
            attn=attn.view(-1,self.num_heads,n,n)
            attn=self.softmax(attn)
        else:
            attn=self.softmax(attn)

        attn=self.attn_drop(attn)
        x=(attn@v).transpose(1,2).reshape(b_,n,c)
        x=self.proj(x)
        x=self.proj_drop(x)
        return x



class SwinTransformerBlcok(nn.Module):
    def __init__(self,dim,num_heads,window_size=7,shift_size=0,
                 mlp_ratio=4.,qkv_bias=True,drop=0.,attn_drop=0.,drop_path=0.,
                 act_layer=nn.GELU,norm_layer=nn.LayerNorm):
        super(SwinTransformerBlcok,self).__init__()
        self.dim=dim
        self.num_heads=num_heads
        self.window_size=window_size
        self.shift_size=shift_size
        self.mlp_ratio=mlp_ratio
        assert 0<=self.shift_size<=self.window_size,'shift size must be 0-window_size'

        self.norm1=norm_layer(dim)
        self.attn=WindowAttention(
            dim,window_size=(self.window_size,-self.window_size),
            num_heads=num_heads,qkv_bias=qkv_bias,
            attn_drop=attn_drop,proj_drop=drop
        )

        self.drop_path=DropPath(drop_path) if drop_path>0 else nn.Identity()
        self.norm2=norm_layer(dim)
        mlp_hidden_dim=int(self.dim*mlp_ratio)
        self.mlp=MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)
    def forward(self,x,attn_mask):
        h,w=self.h,self.w #blk中的hw
        b,l,c=x.shape
        assert l==h*w

        shortcut=x
        x=self.norm1(x)
        x=x.view(b,h,w,c)

        #进行padding
        pad_l=pad_t=0
        pad_r=(self.window_size-w%self.window_size)%self.window_size
        pad_b=(self.window_size-h%self.window_size)%self.window_size
        x=F.pad(x,(0,0,pad_l,pad_r,pad_t,pad_b))
        _,hp,wp,_=x.shape

        if self.shift_size>0:
            #进行sw msa
            #移动一些window
            shifted_x=torch.roll(x,shifts=(-self.shift_size,-self.shift_size),dims=(1,2))
        else:
            shifted_x=x
            attn_mask=None

        #partition windows
        x_windows=window_partition(shifted_x,self.window_size)
        x_windows=x_windows.view(-1,self.window_size*self.window_size,c)

        attn_windows=self.attn(x_windows,mask=attn_mask)

        attn_windows=attn_windows.view(-1,self.window_size,self.window_size,c)
        shifted_x=window_reverse(attn_windows,self.window_size,hp,wp)

        #reverse cyclic shift
        if self.shift_size>0:
            x=torch.roll(shifted_x,shifts=(-self.shift_size,-self.shift_size),dims=(1,2))
        else:
            x=shifted_x

        if pad_r>0 or pad_b>0:
            #把前面pad的数据移除掉
            x=x[:,:h,:w,:].contiguous()
        x=x.view(b,h*w,c)

        x=shortcut+self.drop_path(x)
        x=x+self.drop_path(self.mlp(self.norm2(x)))

        return x
class BasicLayer(nn.Module):
    def __init__(self,dim,depth,num_heads,window_size,mlp_ratio=4.,qkv_bias=True,
                 drop=0.,attn_drop=0.,drop_path=0.,norm_layer=nn.LayerNorm,downsample=None,use_checkpoint=False):
        super(BasicLayer, self).__init__()
        '''
        dim: number of input channels
        depth：每一个stage的swin transformer block数量
        num_heads: number of attention heads
        window_size: local window size
        '''
        self.dim=dim
        self.depth=depth
        self.window_size=window_size
        self.use_checkpoint=use_checkpoint
        self.shift_size=window_size//2 #向右和向下移动的数量

        #当前stage的swintransformerblock的数量
        self.blocks=nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i%2==0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path,list) else drop_path,
                norm_layer=norm_layer
            )for i in range(depth)
        ])
        #是否有patch merging
        if downsample is not  None:
            self.downsample=downsample(dim=dim,norm_layer=norm_layer)
        else:
            self.downsample=None
    def create_mask(self,x,h,w):
        #防止传入hw不是windowsize整数倍
        hp=int(np.ceil(h/self.window_size))*self.window_size #向上取整
        wp=int(np.ceil(w/self.window_size))*self.window_size

        img_mask=torch.zeros((1,hp,wp,1),device=x.device)#[1,hp,wp,1]
        h_slices=(slice(0,-self.window_size),slice(-self.window_size,-self.shift_size),slice(-self.window_size,None))
        w_slices=(slice(0,-self.window_size),slice(-self.window_size,-self.shift_size),slice(-self.window_size,None))
        #通过切片取window的范围

        cnt=0
        for h in h_slices:
            for w in w_slices:
                img_mask[:,h,w,:]=cnt
                cnt+=1
        #相同数字对应的是同一个window
        mask_windows=window_partition(img_mask,self.window_size)#把img_mask切实地划分成一个个窗口
        #[b*num_window,h,w,1]
        mask_windows=mask_windows.view(-1,self.window_size*self.window_size)#[b*num_window,hw]
        attn_mask=mask_windows.unsqueeze(1)-mask_windows.unsqueeze(2) #[b*num_window,1,hw]-[b*num_window,hw,1] 广播机制相减

        attn_mask=attn_mask.masked_fill(attn_mask!=0,float(-100.0)).masked_fill(attn_mask==0,float(0.0)) #不为0表示不属于当前window需要置为无穷小
        return attn_mask

    def forward(self,x,h,w):
        attn_mask=self.creat_mask(x,h,w) #设置swmsa的掩码
        for blk in self.blocks:
            blk.h,blk.w=h,w #添加高宽属性
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x=checkpoint.checkpoint(blk,x,attn_mask)#默认不会使用
        if self.downsample is not None:
            x=self.downsample(x,h,w)#patch merging
            h,w=(h+1)//2,(w+1)//2#新的hw大小 ，为了防止hw为奇数所以+1
        return x,h,w


class SwinTransformer(nn.Module):
    def __init__(self,patch_size=4,in_channels=3,num_classes=1000,
                 embed_dim=96,depth=(2,2,6,2),num_heads=(3,6,12,24),
                 window_size=7,mlp_ratio=4,qkv_bias=True,drop_rate=0.,
                 attn_drop_rate=0.,drop_path_rate=0.1,norm_layer=nn.LayerNorm,
                 patch_norm=True,use_checkpoint=False,**kwargs):
        super(SwinTransformer,self).__init__()
        '''
        patch_size: 对应patch embedding下采样倍数
        embed_dim: linear embedding后的维度
        depth: 每个stage中的block数量
        num_heads: 每个stage的encoder block的head数量
        window_size: 对应WMSA SWMSA的window大小
        mlp_raio: MLP的第一个linear把维度扩大
        '''
        self.num_classes = num_classes
        self.num_layers=len(depth)#stages的个数
        self.embed_dim=embed_dim
        self.patch_norm=patch_norm
        #stage4输出特征矩阵的channels，如4个stage，最后的channels就是8*embed_dim
        self.num_features=int(embed_dim*2**(self.num_layers-1))
        self.mlp_ratio=mlp_ratio

        #split image into non-overlapping patches
        self.patch_embed=PatchEmbedding(patch_size,in_channels,self.embed_dim,norm_layer=norm_layer if self.patch_norm==True else None)
        self.pos_drop=nn.Dropout(p=drop_rate)

        #stochastic depth:dropout rate慢慢增长
        dpr=[x.item() for x in torch.linspace(0,drop_path_rate,sum(depth))]
        self.layers=nn.ModuleList()
        for i_layer in range(self.num_layers):
            #这里每个stage包含的是该stage的swin transformer block+下一个stage的patch merging
            layers=BasicLayer(dim=int(embed_dim*2**i_layer),depth=depth[i_layer],
                              num_heads=num_heads[i_layer],window_size=window_size,mlp_ratio=self.mlp_ratio,qkv_bias=qkv_bias,
                              drop=drop_rate,attn_drop=attn_drop_rate,drop_path=drop_path_rate,norm_layer=norm_layer,downsample=PatchMerging if (i_layer<self.num_layers-1) else None,
                              use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        self.norm=norm_layer(self.num_features)#stage4之后的正则化层
        self.avgpool=nn.AvgPool1d(1)
        self.head=nn.Linear(in_features=self.num_features,out_features=self.num_classes) if self.num_classes>1 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            nn.init.trunc_normal_(m.weight,std=0.02)
            if isinstance(m,nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1.0)

    def forward(self,x):
        x,h,w=self.patch_embed(x)#对图像下采样4倍
        #输出的x的维度是[b,h*w,c]
        x=self.pos_drop(x)

        for layer in self.layers:
            x,h,w=layer(x,h,w)

        x=self.norm(x)
        x=self.avgpool(x.transpose(1,2)) #得到输出[b,c,1]
        x=torch.flatten(x,1)#[b,c]
        x=self.head(x)
        return  x

def swin_tiny_patch4_window7_224(num_classes=1000,**kwargs):
    model=SwinTransformer(in_channels=3,patch_size=4,window_size=7,
                          embed_dim=96,depth=(2,2,6,2),num_heads=(3,6,12,24),num_classes=num_classes,**kwargs)
    return model