import torch
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as tfs
from torch.utils.data import DataLoader, sampler
import numpy as np
import torchvision as tv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (10.0, 10.0)  # 调整生成图的最大尺寸
plt.rcParams['image.interpolation'] = 'nearest'  # 设置差值方式
plt.rcParams['image.cmap'] = 'gray'  # 灰度空间


def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])
    print("images:    ", images.shape[0])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    plt.figure(figsize=(sqrtn, sqrtn))
    print("sqrtn", sqrtn)
    gs = gridspec.GridSpec(sqrtn, sqrtn)  # 分为sqrtn行sqrtn列
    gs.update(wspace=0.05, hspace=0.05)
    num = 0
    for i, img in enumerate(images, 0):
        print("num:  ", num)
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.savefig('fig.png', bbox_inches='tight')
        # plt.imshow(img.reshape([sqrtimg, sqrtimg]))
        num += 1
        return


def preprocess_img(x):
    x = tfs.ToTensor()(x)
    return (x - 0.5)/0.5


def deprocess_img(x):
    return (x + 1.0)/2.0


# 定义取样函数
class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start


    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))


    def __len__(self):
        return self.num_samples


NUM_TRAIN = 50000
NUM_VAL = 5000

NOISE_DIM = 96
batch_size = 128

train_set = tv.datasets.MNIST('./datasets/mnist/',
                  train=True,
                  download=True,
                  transform=preprocess_img
)
train_data = torch.utils.data.DataLoader(train_set,
                        batch_size=batch_size,
                        sampler=ChunkSampler(NUM_TRAIN, 0)  # 从0位置开始采样NUM_TRAIN个数
)

val_set = tv.datasets.MNIST('./datasets/mnist/',
                  train=True,
                  download=True,
                  transform=preprocess_img
)

val_data = torch.utils.data.DataLoader(val_set,
                        batch_size=batch_size,
                        sampler=ChunkSampler(NUM_VAL, NUM_TRAIN)  # 从NUM_TRAIN位置开始采样NUM_VAL个数
)


imgs = deprocess_img(train_data.__iter__().next()[0].view(batch_size, 784)).numpy().squeeze()
show_images(imgs)
# print("imgs", imgs)
# print("train_data.__iter__().next():              ", train_data.__iter__().next())
# print("train_data.__iter__().next()[0]:              ", train_data.__iter__().next()[0].shape)  # train_data.__iter__().next()[0].shape torch.Size([128, 1, 28, 28])
# print("train_data.__iter__().next()[0].view(batch_size, 784):              ", train_data.__iter__().next()[0].view(batch_size, 784))



# 判断网络
def discriminator():
    net = nn.Sequential(
        nn.Linear(784, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1)
    )
    return net


# 生成网络
def generator(noise_dim=NOISE_DIM):
    net = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 784),
        nn.Tanh()
    )
    return net


bce_loss = nn.BCEWithLogitsLoss()  # 先对输出向量里的每个元素使用sigmoid函数, 然后使用BCELoss函数


# 判别器的损失函数
def discriminator_loss(logits_real, logits_fake):
    # print('logits_real:   ', logits_real)
    # print('logits_fake:   ', logits_fake)
    size = logits_real.shape[0]  # 128
    true_labels = Variable(torch.ones(size, 1)).float()  # 128行1列 且元素全为1 print('判别器的损失函数true_labels', true_labels.shape)
    false_labels = Variable(torch.ones(size, 1)).float()
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, false_labels)  # 真实图片识别 虚假图片识别
    return loss


# 生成器的损失函数
def generator_loss(logits_fake):
    size = logits_fake.shape[0]
    true_labels = Variable(torch.ones(size, 1)).float()
    loss = bce_loss(logits_fake, true_labels)  # 假图片的生成
    return loss


# 优化函数
def get_optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    return optimizer


def train_a_gan(D_net, G_net, D_optimizer, G_optimizer ,discriminator_loss, generator_loss,
                show_every=256, noise_size=96, num_epochs=1):
    iter_count = 0
    for epoch in range(num_epochs):  # 训练10次
        print("第：{} 次训练".format(epoch))
        for x, _ in train_data:
            bs = x.shape[0]  # x为 torch.Size([128, 1, 28, 28])      _ 为torch.Size([128])
            # 判别网络
            real_data = Variable(x).view(bs, -1)  # bs为128 -1标识不确定
            logits_real = D_net(real_data)  # 判别器网络训练 对于真实图片

            sample_noise = (torch.rand(bs, noise_size) - 0.5)/0.5
            g_fake_seed = Variable(sample_noise)  # 假图片的种子---噪声
            fake_images = G_net(g_fake_seed)  # G生成假图片fake_images 生成器网络对于假图片训练

            logits_fake = D_net(fake_images)  # D来判断这些假图片  训练D

            d_total_error = discriminator_loss(logits_real, logits_fake)  # 判别器 对于真实图片与生成的假图片的判断 损失函数
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step()

            # 生成网络
            g_fake_seed = Variable(sample_noise)
            fake_images = G_net(g_fake_seed)  # 生成器网络进行训练 生成假的数据
            gen_logits_fake = D_net(fake_images)  # 判别器对于假图片进行判别
            g_error = generator_loss(gen_logits_fake)  # 生成器的损失函数
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step()

            if(iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G: {:.4}'.format(iter_count, d_total_error.item(), g_error.item()))  # item()一个元素张量可以用item得到元素值
                imgs_numpy = deprocess_img(fake_images.data.cpu().numpy())  # (128, 784)
                show_images(imgs_numpy[0:16])   # (16, 784)
                plt.show()
                print()
            iter_count += 1


D = discriminator()
G = generator()

D_optim = get_optimizer(D)
G_optim = get_optimizer(G)

train_a_gan(D, G, D_optim, G_optim, discriminator_loss, generator_loss)







