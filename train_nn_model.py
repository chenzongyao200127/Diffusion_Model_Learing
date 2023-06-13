# Lab 2 Training

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from IPython.display import HTML
from diffusion_utilities import *

import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import PillowWriter # 导入 PillowWriter 用于保存 GIF 文件

# Setting Things Up
class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):  # cfeat - context features
        super(ContextUnet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height  #assume h == w. must be divisible by 4, so 28,24,20,16...

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)        # down1 #[10, 256, 8, 8]
        self.down2 = UnetDown(n_feat, 2 * n_feat)    # down2 #[10, 256, 4,  4]
        
         # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4), # up-sample  
            nn.GroupNorm(8, 2 * n_feat), # normalize                       
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1), # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(8, n_feat), # normalize
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1), # map to same number of channels as input
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """
        # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

        # pass the input image through the initial convolutional layer
        x = self.init_conv(x)
        # pass the result through the down-sampling path
        down1 = self.down1(x)       #[10, 256, 8, 8]
        down2 = self.down2(down1)   #[10, 256, 4, 4]
        
        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)
        
        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
            
        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)     # (batch, 2*n_feat, 1,1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        #print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")


        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

# 超参数

# hyperparameters

# diffusion hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 64 # 64 hidden dimension feature
n_cfeat = 5 # context vector is of size 5
height = 16 # 16x16 image
save_dir = './weights/'

# training hyperparameters
batch_size = 100
n_epoch = 32
lrate=1e-3


# 构造DDPM噪声调度
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1


# 构造模型
nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)


# Training

# load dataset and construct optimizer
# 加载数据集和构建优化器
# 首先，从磁盘上加载一个自定义数据集，数据集包含图像和标签。
# 然后，创建一个DataLoader对象，用于在训练过程中从数据集中加载批量数据。
# 接着，使用Adam优化器来优化神经网络模型（nn_model）的参数。
dataset = CustomDataset("./sprites_1788_16x16.npy", "./sprite_labels_nc_1788_16x16.npy", transform, null_context=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

# helper function: perturbs an image to a specified noise level
# 定义扰动函数
# 定义一个名为perturb_input的函数，用于在输入图像和噪声之间添加扰动。
# 这个函数接受输入图像x、时间步t和噪声noise作为参数，返回扰动后的图像。
'''
实际上，当将噪声应用到输入图像上时，perturb_input函数会考虑时间步t。
这是通过将噪声与输入图像的权重线性插值来实现的。
具体来说，当t较小时，噪声所占的权重较小；当t较大时，噪声所占的权重较大。
'''
def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise


# training without context code

# set into train mode
nn_model.train()

for ep in range(n_epoch):
    # 打印当前周期数。
    print(f'epoch {ep}')
    
    # linearly decay learning rate
    # 线性衰减学习率。
    optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
    
    # 使用tqdm模块创建一个进度条，用于展示批量数据处理的进度。
    pbar = tqdm(dataloader, mininterval=2 )
    
    # 遍历数据加载器生成的批量数据（图像x）。
    for x, _ in pbar:   # x: images
        # 清除优化器的梯度缓存。
        optim.zero_grad()
        
        # 将图像移动到指定的设备（例如GPU）。
        x = x.to(device)
        
        # perturb data
        # 生成与输入图像具有相同形状的随机噪声。
        noise = torch.randn_like(x)
        
        # 随机选择一个时间步t。
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device) 
        
        # 使用perturb_input函数扰动输入图像
        x_pert = perturb_input(x, t, noise)
        
        # use network to recover noise
        # 将扰动后的图像输入神经网络，预测噪声。
        pred_noise = nn_model(x_pert, t / timesteps)
        
        # loss is mean squared error between the predicted and true noise
        # 计算预测噪声与真实噪声之间的均方误差作为损失。
        loss = F.mse_loss(pred_noise, noise)
        
        # 反向传播误差以计算梯度。
        loss.backward()
        
        # 使用优化器更新神经网络的权重
        optim.step()

    # save model periodically
    # 定期保存模型：在每个训练周期后，根据条件（例如每4个周期或在最后一个周期）将模型的状态字典保存到磁盘上。
    # 如果保存目录不存在，则创建该目录。
    if ep%4==0 or ep == int(n_epoch-1):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(nn_model.state_dict(), save_dir + f"model_{ep}.pth")
        print('saved model at ' + save_dir + f"model_{ep}.pth")
        
        
# Sampling
# helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

# sample using standard algorithm
@torch.no_grad()
def sample_ddpm(n_sample, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, height, height).to(device)  

    # array to keep track of generated steps for plotting
    intermediate = [] 
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t)    # predict noise e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate ==0 or i==timesteps or i<8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

        
'''
 View Epochs
'''
# View Epoch 0
# load in model weights and set to eval mode
nn_model.load_state_dict(torch.load(f"{save_dir}/model_0.pth", map_location=device))
nn_model.eval()
print("Loaded in Model")

# 清除当前的图形，以便开始绘制新的图形
plt.clf()
samples, intermediate_ddpm = sample_ddpm(32)
animation_ddpm = plot_sample(intermediate_ddpm, 32, 4, save_dir, "ani_run_epoch_0", None, save=True)
gif_writer = PillowWriter(fps=24) # 设置帧率为 24 fps
animation_ddpm.save('animation_ddpm.gif', writer=gif_writer)



# View Epoch 4
# load in model weights and set to eval mode
nn_model.load_state_dict(torch.load(f"{save_dir}/model_4.pth", map_location=device))
nn_model.eval()
print("Loaded in Model")

# 清除当前的图形，以便开始绘制新的图形
plt.clf()
samples, intermediate_ddpm = sample_ddpm(32)
animation_ddpm = plot_sample(intermediate_ddpm, 32, 4, save_dir, "ani_run_epoch_4", None, save=True)
gif_writer = PillowWriter(fps=24) # 设置帧率为 24 fps
animation_ddpm.save('animation_ddpm.gif', writer=gif_writer)



# View Epoch 8
# load in model weights and set to eval mode
nn_model.load_state_dict(torch.load(f"{save_dir}/model_8.pth", map_location=device))
nn_model.eval()
print("Loaded in Model")

# 清除当前的图形，以便开始绘制新的图形
plt.clf()
samples, intermediate_ddpm = sample_ddpm(32)
animation_ddpm = plot_sample(intermediate_ddpm, 32, 4, save_dir, "ani_run_epoch_8", None, save=True)
gif_writer = PillowWriter(fps=24) # 设置帧率为 24 fps
animation_ddpm.save('animation_ddpm.gif', writer=gif_writer)


# View Epoch 31
# load in model weights and set to eval mode
nn_model.load_state_dict(torch.load(f"{save_dir}/model_31.pth", map_location=device))
nn_model.eval()
print("Loaded in Model")


# 清除当前的图形，以便开始绘制新的图形
plt.clf()
samples, intermediate_ddpm = sample_ddpm(32)
animation_ddpm = plot_sample(intermediate_ddpm, 32, 4, save_dir, "ani_run_epoch_31", None, save=True)
gif_writer = PillowWriter(fps=24) # 设置帧率为 24 fps
animation_ddpm.save('animation_ddpm.gif', writer=gif_writer)