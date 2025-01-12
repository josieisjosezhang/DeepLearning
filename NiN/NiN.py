import torch
import torch.nn as nn
import logging
import os
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter
import json

#配置
parser=argparse.ArgumentParser()
parser.add_argument("--output_dir",type=str,default='outs',help='the dir that you want to put your result')
parser.add_argument("--device",type=str,default='cuda',help="choose your device")
parser.add_argument("--lr",type=float,default=0.0001,help="the learning rate")
parser.add_argument("--log",type=str,default='logs',help='the name of log.txt')
parser.add_argument("--momentum",type=float,default=0.5,help="momentum")
parser.add_argument("--batchsize",type=int,default=2,help="batchsize")
parser.add_argument("--epoch",type=int,default=25,help='epoch')
opt=parser.parse_args()
print(opt)
dir=os.makedirs(f"NiN/{opt.output_dir}",exist_ok=True)
def get_logger(filename,verbosity=1,name=None):
    level={0:logging.DEBUG,1:logging.INFO,2:logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][line:%(lineno)d][%(levelname)s]%(message)s")
    logger=logging.getLogger(name)
    logger.setLevel(level[verbosity])
    fh=logging.FileHandler(filename,'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh=logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
log=get_logger(f'NiN/{opt.log}.txt')
#
writer1=SummaryWriter(f'NiN/{opt.output_dir}')
writer2=SummaryWriter(f'NiN/{opt.output_dir}')
#
content=opt
content=vars(content)
file=r'NiN/config.json'
with open(file,'w',encoding='utf-8') as f:
    json.dump(content,f,indent=4,ensure_ascii=True)
#
if opt.device=='cuda' and torch.cuda.is_available()==True:
    use_device='cuda'
else:
    use_device='cpu'
#数据
transforms=transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
train=torchvision.datasets.FashionMNIST(root='data/FashionMNIST',train=True,transform=transforms,download=True)
val=torchvision.datasets.FashionMNIST(root='data/FashionMNIST',train=False,transform=transforms,download=True)
train=DataLoader(train,batch_size=opt.batchsize,shuffle=True)
val=DataLoader(val,batch_size=opt.batchsize,shuffle=True)
#网络
def block(inchannel,outchannel,size,strides,pad):
    layers=[]
    layers.append(nn.Conv2d(inchannel,outchannel,kernel_size=size,stride=strides,padding=pad))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(in_channels=outchannel,out_channels=outchannel,kernel_size=1))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(in_channels=outchannel,out_channels=outchannel,kernel_size=1))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)
conv_arch=[(96,11,4,0),(256,5,1,1),(384,3,1,1),(10,3,1,1)]
def net(conv_arch):
    inchannel=1
    net=[]
    for i in range(len(conv_arch)-1):
        out, size, strides, pad=conv_arch[i]
        net.append(block(inchannel,out,size,strides,pad))
        net.append(nn.MaxPool2d(kernel_size=3,stride=2))
        inchannel=out
    out, size, strides, pad = conv_arch[-1]
    net.append(nn.Dropout(0.5))
    net.append(block(inchannel,out, size, strides, pad))
    net.append(nn.AdaptiveAvgPool2d((1,1)))
    net.append(nn.Flatten())
    return nn.Sequential(*net)
nin=net(conv_arch).to(torch.device(use_device))
#输出网络情况
# X = torch.randn(size=(1, 1, 224, 224))
# for blk in nin:
#     X = blk(X)
#     print(blk.__class__.__name__,'output shape:\t',X.shape)

#loss
loss_func=nn.CrossEntropyLoss()
#optim
optim=torch.optim.Adam(nin.parameters(),lr=opt.lr)
#训练
def trainer(epoch):
    right=0
    loss=0
    sample=0
    for i,(x,y) in enumerate(train):
        x=x.to(torch.device(use_device))
        y= y.to(torch.device(use_device))
        y_hat = nin(x)
        cost=loss_func(y_hat,y)
        optim.zero_grad()
        cost.backward()
        optim.step()
        with torch.no_grad():
            loss+=cost.data.sum()
            right+=(torch.argmax(y_hat,dim=1)==y).sum()
            sample+=y.size(0)
    train_loss=loss / len(train)
    train_acc=right/sample*100
    log.info(f'[train]\t[epoch:{epoch + 1}/{opt.epoch}]\tloss={train_loss:.3f}\tacc={train_acc:.3f}%')
    return train_loss,train_acc
#验证
def valer():
    right=0
    loss=0
    sample=0
    with torch.no_grad():
        for i,(x,y) in enumerate(val):
            x = x.to(torch.device(use_device))
            y = y.to(torch.device(use_device))
            y_hat=nin(x)
            cost=loss_func(y_hat,y)
            loss+=cost.data.sum()
            right+=(torch.argmax(y_hat,dim=1)==y).sum()
            sample+=y.size(0)
        val_loss = loss / len(val)
        val_acc = right / sample * 100
        log.info(f'[val]\t[epoch:{epoch + 1}/{opt.epoch}]\tloss={val_loss:.3f}\tacc={val_acc:.3f}%')
        return val_loss, val_acc

for epoch in range(opt.epoch):
    train_loss,train_acc=trainer(epoch)
    val_loss, val_acc=valer()
    writer1.add_scalars('acc',{'train':train_acc,'val':val_acc},epoch+1)
    writer2.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch + 1)
pt=torch.save(nin.state_dict(),'NiN/outs/net.pt')
net_copy=net(conv_arch).to(torch.device(use_device))
net_copy.load_state_dict(torch.load('NiN/outs/net.pt'))
