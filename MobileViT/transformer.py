import torch
import torch.nn as nn
import numpy as np

__all__ = ['TransformerEncoder']


class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,attn_drop_ratio,bias):
        super(MultiHeadAttention, self).__init__()
        self.qkv=nn.Linear(embed_dim,3*embed_dim,bias=bias)
        self.num_heads=num_heads

        self.scale=np.sqrt(embed_dim)
        self.softmax=nn.Softmax(dim=-1)
        self.attn_drop=nn.Dropout(attn_drop_ratio)

        self.proj=nn.Linear(embed_dim,embed_dim,bias=bias)
    def forward(self,x):
        b,n,c = x.size()
        #[b,n,c]->[b,n,3c]->[b,n,3,num_head,head_dim]->[3,b,n,num_head,head_dim]
        qkv = self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,1,3,4).contiguous()
        #[1,b,n,num_head,head_dim]->[b,num_head,n,head_dim]
        q, k, v = qkv[0].reshape(b,self.num_heads,n,-1), qkv[1].reshape(b,self.num_heads,n,-1),qkv[2].reshape(b,self.num_heads,n,-1)

        attn=q*self.scale@k.transpose(-2,-1)
        attn=self.softmax(attn)
        attn=self.attn_drop(attn)
        attn=attn@v
        attn=attn.reshape(b,n,c)
        attn=self.proj(attn)
        return attn



class TransformerEncoder(nn.Module):
    def __init__(self,embed_dim,ffn_latent_dim,num_heads,attn_drop_ratio,dropout,ffn_drop_ratio):
        super(TransformerEncoder, self).__init__()

        attn_unit=MultiHeadAttention(embed_dim,num_heads,attn_drop_ratio,bias=True)
        self.pre_norm_mhsa=nn.Sequential(
            nn.LayerNorm(embed_dim),
            attn_unit,
            nn.Dropout(p=dropout)
        )

        self.pre_norm_ffn=nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim,ffn_latent_dim,bias=True),
            nn.SiLU(),
            nn.Dropout(p=ffn_drop_ratio),
            nn.Linear(ffn_latent_dim,embed_dim,bias=True),
            nn.Dropout(p=ffn_drop_ratio)
        )

        self.embed_dim=embed_dim
        self.ffn_dim=ffn_latent_dim
        self.ffn_drop_ratio=ffn_drop_ratio
        self.std_drop=dropout
    def forward(self,x):
        x = self.pre_norm_mhsa(x)+x
        x = self.pre_norm_ffn(x)+x
        return x
