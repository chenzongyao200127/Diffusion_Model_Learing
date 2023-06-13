# Lab 1, Sampling

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

# 扩散超参数
timesteps = 500
beta1 = 1e-4
beta2 = 0.02


# 网络超参数
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
print("torch.device", device)
n_feat = 64 # 64个隐藏维度特征
n_cfeat = 5 # 上下文向量大小为5
height = 16 # 16x16图像
save_dir = './weights/'

# 构造DDPM噪声调度
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

# 构造模型
nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

# Sampling
'''
这段代码定义了一个名为denoise_add_noise的辅助函数，用于在去噪过程中添加噪声。
接着，它从文件加载训练好的模型权重，并将模型设置为评估模式。
x：输入图像张量
t：当前时间步
pred_noise：预测的噪声
z（可选）：噪声张量；如果未给定，将生成一个与输入图像相同形状的随机噪声张量

函数的主要目的是在去噪过程中添加噪声，以避免模型崩溃。
它首先计算噪声的缩放因子，然后计算去噪后的均值。最后，它返回去噪后的均值加上噪声。

接下来，代码从文件加载训练好的模型权重，并将模型设置为评估模式。
这是因为在采样阶段，我们不希望改变模型的权重。
将模型设置为评估模式还可以关闭某些特定于训练阶段的行为，例如Dropout层和Batch Normalization层。
'''
# 辅助函数；去除预测的噪声（但添加一些噪声以避免模型崩溃）
def denoise_add_noise(x, t, pred_noise, z=None):
    # 如果没有给定噪声张量 z，则生成一个与 x 相同形状的随机噪声张量
    if z is None:
        z = torch.randn_like(x)
    # 计算噪声的缩放因子
    noise = b_t.sqrt()[t] * z
    # 计算去噪后的均值
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    # 返回去噪后的均值加上噪声
    return mean + noise

# 从文件加载训练好的模型权重
nn_model.load_state_dict(torch.load(f"{save_dir}/model_trained.pth", map_location=device))
# 将模型设置为评估模式
nn_model.eval()
print("Loaded in Model")

'''
这段代码定义了一个名为sample_ddpm的函数，用于使用标准算法采样。该函数利用训练好的模型和DDPM算法生成一组图像样本。
    sample_ddpm函数接受以下参数：
    - n_sample：要生成的样本数量
    - save_rate（可选）：保存中间结果的频率，默认为每20个时间步保存一次
    
    函数首先从标准正态分布（均值为0，标准差为1）中采样初始噪声。
    然后，它从最后一个时间步开始，逐步向前采样。
    在每个时间步，它调整时间张量的形状，然后根据需要采样一些随机噪声。
    接下来，它使用训练好的模型预测噪声，并使用denoise_add_noise辅助函数去除预测的噪声并添加一些噪声。
    最后，按给定的保存频率保存中间结果，以便于绘制动画。返回的是最终样本和中间结果。
'''
# 使用标准算法进行采样
@torch.no_grad()
def sample_ddpm(n_sample, save_rate=20):
    # 从 N(0, 1) 分布中采样初始噪声 x_T
    samples = torch.randn(n_sample, 3, height, height).to(device)

    # 用于跟踪生成步骤的数组，以便于绘制
    intermediate = []
    # 从最后一个时间步开始，逐步向前采样
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # 调整时间张量的形状
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # 采样一些随机噪声以注入回。对于 i = 1，不添加噪声
        z = torch.randn_like(samples) if i > 1 else 0

        # 预测噪声 e_(x_t,t)
        eps = nn_model(samples, t)
        
        # 使用辅助函数去除预测的噪声并添加一些噪声
        samples = denoise_add_noise(samples, i, eps, z)
        
        # 按给定的保存频率保存中间结果，便于绘制动画
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    # 将中间结果堆叠成一个 NumPy 数组
    intermediate = np.stack(intermediate)
    return samples, intermediate


# # 清除当前的图形，以便开始绘制新的图形
# plt.clf()

# # 从DDPM (denoising diffusion probabilistic models) 中采样 32 个样本
# # 并获得采样过程的中间结果
# samples, intermediate_ddpm = sample_ddpm(32)

# # 使用 plot_sample 函数生成动画，展示 32 个样本在采样过程中的变化
# # 我们将动画保存在 save_dir 目录下，文件名为 "ani_run"
# # save 参数设置为 False，表示不保存动画文件
# animation_ddpm = plot_sample(intermediate_ddpm, 32, 4, save_dir, "ani_run", None, save=False)

# # 将动画转换为 HTML 格式，并在 Jupyter Notebook 或其他支持的环境中显示
# HTML(animation_ddpm.to_jshtml())





# 清除当前的图形，以便开始绘制新的图形
plt.clf()

# 从DDPM (denoising diffusion probabilistic models) 中采样 32 个样本
# 并获得采样过程的中间结果
samples, intermediate_ddpm = sample_ddpm(32)

# 使用 plot_sample 函数生成动画，展示 32 个样本在采样过程中的变化
# 我们将动画保存在 save_dir 目录下，文件名为 "ani_run"
# save 参数设置为 True，表示保存动画文件
animation_ddpm = plot_sample(intermediate_ddpm, 32, 4, save_dir, "ani_run", None, save=True)

# 保存动画为 GIF 文件
gif_writer = PillowWriter(fps=24) # 设置帧率为 24 fps
animation_ddpm.save('animation_ddpm.gif', writer=gif_writer)

# # 保存为 MP4 文件
# from matplotlib.animation import FFMpegWriter
# animation_ddpm.save('animation_ddpm.mp4', writer=FFMpegWriter(fps=24))


# 在DDPM（Denoising Diffusion Probabilistic Models）算法中，额外的噪声（z）在反向采样过程中起着关键作用。
# 该算法通过从一个噪声图像开始，逐渐去除噪声以生成所需的样本。
# 在这个过程中，噪声的缩减是通过预测每个时间步的噪声（eps）来实现的。
# 然而，仅仅依靠预测的噪声去除可能无法完全还原到原始分布。

# 添加额外的噪声z对于探索目标分布的不同区域和生成多样性更丰富的图像至关重要。
# 在反向采样过程中，额外的噪声z有助于在每个时间步引入一定程度的随机性，从而使得生成的图像更接近于原始分布。
# 这种随机性确保了生成的样本不会过于聚焦于局部特征，而是能够覆盖整个目标分布。

# 当我们不添加额外的噪声（z = 0）时，我们实际上限制了模型探索目标分布的能力。
# 这可能导致生成的图像过于简化，丢失原始分布中的许多重要特征和细节。
# 因此，为了生成与原始分布更接近的图像，添加额外的噪声z至关重要。

# Demonstrate incorrectly sample without adding the 'extra noise'
# 错误地演示不添加“额外噪声”的采样
@torch.no_grad()
def sample_ddpm_incorrect(n_sample):
    # 从 N(0, 1) 分布中采样初始噪声 x_T
    samples = torch.randn(n_sample, 3, height, height).to(device)

    # 用于跟踪生成步骤的数组，以便于绘制
    intermediate = []
    # 从最后一个时间步开始，逐步向前采样
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # 调整时间张量的形状
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # 不添加噪声
        z = 0

        # 预测噪声 e_(x_t,t)
        eps = nn_model(samples, t)
        # 在DDPM算法中，噪声缩减的实现主要依赖于训练好的神经网络模型。
        # 这个模型被训练来预测在每个时间步t的噪声值eps。
        # 这些噪声值表示为条件高斯噪声，即给定当前状态x_t和时间t时，原始图像x_0所受到的噪声
        
        # 在反向采样过程中，神经网络模型接收当前的图像样本x_t和时间t作为输入。
        # 对于每个时间步，模型预测出对应的噪声值eps：
        
        # eps = nn_model(samples, t)
        
        # 这里的samples表示在当前时间步t的图像样本（即x_t），
        # 而eps表示从x_t到x_{t-1}的噪声值。
        # 然后，我们使用一个辅助函数denoise_add_noise来执行去噪和添加噪声的操作。
        # 这个辅助函数首先从当前样本x_t中去除预测的噪声值eps，然后再添加额外的噪声z：
        
        # samples = denoise_add_noise(samples, i, eps, z)
        
        # 这个操作将图像样本从当前时间步t更新到时间步t-1，逐步减少噪声。
        # 通过在整个时间步序列中重复这个过程，
        # 我们可以从最后一个时间步（最嘈杂的图像）开始，逐步去除噪声，最终生成与原始分布更接近的图像。
        
        
        # 使用辅助函数去除预测的噪声，但不添加噪声
        samples = denoise_add_noise(samples, i, eps, z)
        if i % 20 == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    # 将中间结果堆叠成一个 NumPy 数组
    intermediate = np.stack(intermediate)
    return samples, intermediate


# # visualize samples
# plt.clf()
# samples, intermediate = sample_ddpm_incorrect(32)
# animation = plot_sample(intermediate,32,4,save_dir, "ani_run", None, save=False)
# HTML(animation.to_jshtml())

# 清除当前的图形，以便开始绘制新的图形
plt.clf()

# 从DDPM (denoising diffusion probabilistic models) 中采样 32 个样本
# 并获得采样过程的中间结果
samples, intermediate = sample_ddpm_incorrect(32)

# 使用 plot_sample 函数生成动画，展示 32 个样本在采样过程中的变化
# 我们将动画保存在 save_dir 目录下，文件名为 "ani_run"
# save 参数设置为 True，表示保存动画文件
animation = plot_sample(intermediate, 32, 4, save_dir, "ani_run_2", None, save=True)

# 保存动画为 GIF 文件
gif_writer = PillowWriter(fps=24) # 设置帧率为 24 fps
animation.save('animation_ddpm_2.gif', writer=gif_writer)



# Acknowledgments¶
# Sprites by ElvGames, FrootsnVeggies and kyrise
# This code is modified from, https://github.com/cloneofsimo/minDiffusion
# Diffusion model is based on Denoising Diffusion Probabilistic Models and Denoising Diffusion Implicit Models