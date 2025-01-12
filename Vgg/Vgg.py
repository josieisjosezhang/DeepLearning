import os
import json
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse

#配置
parser=argparse.ArgumentParser()
parser.add_argument("--device",type=str,default='cuda',help='choose cpu or cuda')
parser.add_argument("--output_dir",type=str,default="Vgg/runs")
parser.add_argument("--epoch",type=int,default=100,help="the number of epoch")
parser.add_argument("--batch_size",type=int,default=2,help="the size of batch")
parser.add_argument("--lr",type=float,default=0.1,help="learning rate")
parser.add_argument("--momentum",type=float,default=0.5,help="momentum")
parser.add_argument("--dropout",type=float,default=0.5,help="dropout rate")
opt=parser.parse_args()
print(opt)
#创建输出文件
dir=os.makedirs(f'{opt.output_dir}',exist_ok=True)
#写logger的函数
def get_logger(filename,verbosity=1,name=None):
    level_dict={0:logging.DEBUG,1:logging.INFO,2:logging.WARNING}
    formatter=logging.Formatter("[%(asctime)s][line:%(lineno)d][%(levelname)s]%(message)s")
    logger=logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh=logging.FileHandler(filename,'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh=logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
log=get_logger(f'{opt.output_dir}/exp.txt')
#配置文件写入
content=opt
content=vars(content)
config=os.path.join(opt.output_dir,'config.json')
with open(config,'w',encoding='utf-8') as f:
    json.dump(content,f,indent=4,ensure_ascii=False)

#设备选择
if opt.device:
    device='cuda' if torch.cuda.is_available() else 'cpu'

#数据
train=torchvision.datasets.FashionMNIST(root='./data',train=True,transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),torchvision.transforms.ToTensor()]),download=True)
val=torchvision.datasets.FashionMNIST(root='./data',train=False,transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),torchvision.transforms.ToTensor()]),download=True)
train=DataLoader(train,batch_size=opt.batch_size,shuffle=True)
val=DataLoader(val,batch_size=opt.batch_size,shuffle=True)

#VGG块
def block(conv_num,in_channel,out_channel):
    layers=[]
    for _ in range(conv_num):
        layers.append(nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1))#只改变通道，不改变图像大小\
        layers.append(nn.ReLU())
        in_channel=out_channel
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))#图像大小减半
    return nn.Sequential(*layers)
#*layers是解包序列，nn.Sequential()，从而构建一个包含这些子模块的序列模型
def vgg(conv_arch):
    net=[]
    inchannel=1
    for (number,outchannel) in conv_arch:
        net.append(block(number,inchannel,outchannel))
        inchannel=outchannel
    return nn.Sequential(*net,nn.Flatten(),nn.Linear(outchannel*7*7,4096),nn.ReLU(),nn.Dropout(opt.dropout),
                         nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(opt.dropout),
                         nn.Linear(4096,10))
#VGG-19的结构
# conv_arch=[(2,64),(2,128),(4,256),(4,512),(4,512)]
#VGG-16
# conv_arch=[(2,64),(2,128),(3,256),(3,512),(3,512)]
#VGG-11
conv_arch=[(1,64//4),(1,128//4),(2,256//4),(2,512//4),(2,512//4)]
#1*224*224--64*112*112--128*56*56--256*28*28--512*14*14--512*7*7
net=vgg(conv_arch).to(torch.device(device))
# net=vgg(conv_arch)
# net.to('cpu')
#优化器、损失函数
loss_func=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=opt.lr,momentum=opt.momentum,weight_decay=0.5)
#
# X = torch.randn(size=(1, 1, 224, 224))
# for blk in net:
#     X = blk(X)
#     print(blk.__class__.__name__,'output shape:\t',X.shape)

#开始训练
def trainer(epcoh):
    run_loss=0
    right=0
    sample=0
    for i,data in enumerate(train):
        x,y=data
        x=x.to(torch.device(device))
        y=y.to(torch.device(device))
        y_hat=net(x)
        loss=loss_func(y_hat,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            run_loss+=loss.data.sum()
            right+=(torch.argmax(y_hat,dim=1)==y).sum()
            sample+=y.size(0)
    acc_train=right/sample*100
    loss_train=run_loss/len(train)
    log.info(f'[train]\t[epoch:{epoch+1}/{opt.epoch}]\tloss={loss_train:.3f}\tacc={acc_train:.3f}%')
    return acc_train,loss_train
def valer():
    run_loss=0
    right=0
    sample=0
    with torch.no_grad():
        for i,data in enumerate(val):
            x,y=data
            x=x.to(torch.device(device))
            y=y.to(torch.device(device))
            y_hat=net(x)
            loss=loss_func(y_hat,y)
            run_loss+=loss.data.sum()
            right+=(torch.argmax(y_hat,dim=1)==y).sum()
            sample+=y.size(0)
        acc_val=right/sample*100
        loss_val=run_loss/len(val)
        log.info(f'[val]\t[epoch:{epoch+1}/{opt.epoch}]\tloss={loss_val:.3f}\tacc={acc_val:.3f}%')
        return acc_val,loss_val
pic1=SummaryWriter('Vgg/runs/logs')
pic2=SummaryWriter('Vgg/runs/logs')
log.info('start training !')
for epoch in range(opt.epoch):
    acc_train,loss_train=trainer(epoch)
    acc_val,loss_val=valer()
    pic1.add_scalars('acc',{'train_acc':acc_train,'val_acc':acc_val},epoch+1)
    pic2.add_scalars('loss', {'train_loss': loss_train, 'val_loss': loss_val}, epoch + 1)
pt=torch.save(net.state_dict(),'Vgg/runs/net.pt')
net_copy=net().to(torch.device(device))
net_copy.load_state_dict(torch.load('Vgg/runs/net.pt'))

