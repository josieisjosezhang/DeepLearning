from functools import partial
import  torch
import torch.nn as nn


def drop_path(x,drop_prob,training):
    if drop_prob ==0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape=(x.shape[0],)+(1,)*(x.ndim-1)
    random_tensor=keep_prob+torch.rand(shape,device=x.device,dtype=x.dtype)
    random_tensor.floor_()
    output=x.div(keep_prob)*random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self,drop_prob=None):
        super(DropPath,self).__init__()
        self.drop_prob = drop_prob

    def forward(self,x):
        return drop_path(x,self.drop_prob,self.training)

class PatchEmbedding(nn.Module):
    def __init__(self,img_size=224,patch_size=16,in_c=3,embed_dim=768,norm_layer=None):
        super(PatchEmbedding, self).__init__()
        img_size=(img_size,img_size)
        patch_size=(patch_size,patch_size)
        self.img_size=img_size
        self.patch_size=patch_size #patch的数量
        self.grid_size=(img_size[0]//patch_size[0],img_size[1]//patch_size[1])#14
        self.num_patches=self.grid_size[0]*self.grid_size[1]

        self.proj=nn.Conv2d(in_c,embed_dim,kernel_size=self.patch_size,stride=self.patch_size)
        self.norm=norm_layer(embed_dim) if norm_layer else nn.Identity() #identity()是不做任何操作

    def forward(self,x):
        b,c,h,w=x.size()
        assert h==self.img_size[0] and w==self.img_size[1] #输入图片大小必须是固定

        #flatten：[b,c,h,w]->[b,c,wh]
        #permute：[b,c,wh]->[b,hw,c]
        x=self.proj(x).flatten(2).permute(0,2,1)#表示从index=2开始展平
        x=self.norm(x)
        return x

class Attention(nn.Module):
    '''MHA attention mechanism'''
    def __init__(self,dim,num_heads=8,qkv_bias=False,qk_scale=None,attn_drop_ratio=0.,proj_drop_ratio=0.):
        super(Attention, self).__init__()
        '''
        dim: 输入token的维度(这里是768，token的个数是14*14)
        num_heads: head的个数
        qkv_bias：生成qkv是否使用偏置
        qk_scale：
        '''
        self.num_heads=num_heads
        head_dim=dim//num_heads
        self.scale = qk_scale or head_dim**-0.5 #也就是sqrt(d_k)
        self.qkv=nn.Linear(dim,dim*3,bias=qkv_bias) #可以通过3个全连接层得到qkv也可以使用一个(dim,3*dim)的全连接生成qkvs
        self.attn_drop=nn.Dropout(attn_drop_ratio)
        self.proj=nn.Linear(dim,dim)#对应于把所有head拼接起来后，计算注意力分数的W^O
        self.proj_drop=nn.Dropout(proj_drop_ratio)

    def forward(self,x):
        #[batchsize,token_num+1,token_dim]
        b,n,c=x.shape
        qkv=self.qkv(x) #[batchsize,token_num+1,3*token_dim]
        qkv=qkv.reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4)#[batchsize,token_num+1,3,num_head,head_dim]->[3,batchsize,num_head,token_num+1,head_dim]
        q,k,v=qkv[0],qkv[1],qkv[2] #切片方式拿到q,k,v的数据 q=[1,batchsize,num_head,tokne_num+1,head_dim]

        attn=(q@k.transpose(-2,-1))*self.scale#因为前面已经划分了head，所以这里的操作是对每个head进行的
        #@是矩阵乘法，多维数据进行@只会对最后两个维度进行乘
        attn=attn.softmax(dim=-1) #表示在每一行进行softmax处理
        attn=self.attn_drop(attn)

        #attn@v：[batchsize,num_head,token_num+1,head_dim]
        #transpose：[batch_size,token_num+1,num_head,head_dim]
        #reshape：[batch_size,token_num+1,token_dim]
        x=(attn@v).transppose(1,2).reshape(b,n,c)
        x=self.proj(x)
        x=self.proj_drop(x)

        return x

class MLP(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop=0.):
        super(MLP, self).__init__()
        out_features=out_features or in_features
        hidden_features=hidden_features or in_features
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act1=act_layer()
        self.fc2=nn.Linear(hidden_features,out_features)
        self.drop=nn.Dropout(drop)

    def forward(self,x):
        x=self.fc1(x)
        x=self.act1(x)
        x=self.fc2(x)
        x=self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block,self).__init__()
        '''
        dim: token_dim 这里是768
        num_heads: MHSA中的head数量
        mlp_ratio: MLP中隐藏层的扩大维度
        drop_ratio: MHSA全连接层用的drop ratio
        '''
        self.norm1=norm_layer(dim)
        self.attn=Attention(dim,num_heads,qkv_bias,qk_scale,attn_drop_ratio,drop_ratio)
        self.drop_path=DropPath(drop_path_ratio) if attn_drop_ratio>0. else nn.Identity()

        self.norm2=norm_layer(dim)
        self.mlp=MLP(dim,int(dim*mlp_ratio),act_layer=act_layer,drop=drop_ratio)

    def forward(self,x):
        x=x+self.drop_path(self.attn(self.norm1(x)))
        x=x+self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self,img_size=224,patch_size=16,dim=768,in_c=3,n_heads=12,mlp_ratio=4,num_classes=1000,depth=12,qkv_bias=True,qk_scale=None,
                 drop_ratio=0.,attn_drop_ratio=0.,drop_path_ratio=0.,embed_layer=PatchEmbedding,norm_layer=None,act_layer=None):
        super(VisionTransformer,self).__init__()
        """
        img_size (int,tuple): image size
        patch_size (int,tuple): patch size
        dim (int): dimension of token
        in_c (int): number of input channels
        """
        self.num_classes=num_classes
        self.num_features=dim
        self.num_tokens=1
        norm_layer=norm_layer or partial(nn.LayerNorm,eps=1e-6) #partial方法传入默认参数
        act_layer=act_layer or nn.GELU

        self.patch_embed=embed_layer(img_size=img_size,patch_size=patch_size,in_c=in_c,embed_dim=dim,norm_layer=norm_layer)
        num_patches=self.patch_embed.num_patches

        self.cls_token=nn.Parameter(torch.zeros(1,1,dim))#构建可训练参数 batch,1,token_dim
        self.pos_embed=nn.Parameter(torch.zeros(1,self.num_tokens+num_patches,dim))
        self.pos_drop=nn.Dropout(drop_path_ratio)

        dpr=[x.item() for x in torch.linspace(0, drop_path_ratio,depth)] #构建等差序列，随着encoder block数量而逐渐增加
        self.blocks=nn.Sequential(*[
            Block(dim,n_heads,mlp_ratio,qkv_bias,qk_scale,drop_ratio,attn_drop_ratio,dpr[i],act_layer)
            for i in range(depth)
        ])
        self.norm=norm_layer(dim)

        self.has_logits=False
        self.pre_logits=nn.Identity()

        self.head=nn.Linear(self.num_features,num_classes) if num_classes>0 else nn.Identity()
        self.head_dist=None

        #weight_init
        nn.init.trunc_normal_(self.pos_embed,std=0.02)
        nn.init.trunc_normal_(self.cls_token,std=0.02)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                  nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(self._init_vit_weights)

    def forward_features(self,x):
        #b,c,h,w->b,n,c
        x=self.patch_embed(x)
        cls_token=self.cls_token.expand(x.shape[0],-1,-1)
        x=torch.cat((cls_token,x),dim=1)#[batch,197,768]

        x=self.pos_drop(x+self.pos_embed)
        x=self.blocks(x)
        return self.pre_logits(x[:,0])#切片取最前面cls_token的数据

    def forward(self,x):
        x=self.forward_features(x)
        x=self.head(x)
        return x
def vit_base_patch16_224(num_classes,has_logits=False):
    model=VisionTransformer(img_size=224,patch_size=16,dim=768,in_c=3,num_classes=num_classes)
    return model
