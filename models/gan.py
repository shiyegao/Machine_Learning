import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import os
import numpy as np


class dataLoader():
    def __init__(self, data, mode, transform=None):
        self.mode = mode
        self.transform = transform

        X_train, y_train = data['X_train'], self.onehot(data['y_train'])
        X_test, y_test = data['X_test'], self.onehot(data['y_test'])
        if self.mode=='train':
            self.X = X_train
            self.y = y_train
        elif self.mode=='test':
            self.X = X_test
            self.y = y_test
        self.num_dataset = len(self.X)

    def onehot(self, y):
        y_mat = np.zeros((y.shape[0], len(np.unique(y))))
        for i in range(y.shape[0]):
            y_mat[i, y[i,0]] = 1
        return y_mat

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.num_dataset
    


class discriminator(nn.Module):
    def __init__(self, args):
        super(discriminator, self).__init__()
        # input: batch_size * channels * 32 * 32
        self.dis = nn.Sequential(
            # batch_size * channels * 32 * 32
            nn.Conv2d(args.channels, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True), 

            # batch_size * 32 * 16 * 16
            nn.Conv2d(32, 128, kernel_size=4, stride=2, padding=1), 
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # batch_size * 128 * 16 * 16
            nn.Conv2d(128, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True), 
        )

        # input: batch_size * 1024 * 4 * 4
        self.output = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
        )
 
    def forward(self, x):
        out = self.dis(x)
        return self.output(out)
 


class generator(nn.Module):
    def __init__(self, args):
        super(generator, self).__init__()

        # input: batch_size * z-dim * 1 * 1
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(args.gan_z_dim, 1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 1024 * 4 * 4
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 512 * 8 * 8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 256 * 16 * 16
            nn.ConvTranspose2d(256, args.channels, kernel_size=4, stride=2, padding=1),
        )

        # args.channels * 128 * 128
        self.output = nn.Tanh()
 
    def forward(self, z):
        img = self.gen(z)
        return self.output(img)


class GAN:
    def __init__(self, args):

        self.args = args
        self.critic_iter = 5
        self.cuda = torch.cuda.is_available()
        self.lambda_term = 10

        self.D = discriminator(args)
        self.G = generator(args)
        if self.cuda:
            self.D = self.D.cuda()
            self.G = self.G.cuda()


    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images,_) in enumerate(data_loader):
                yield images


    def calculate_gradient_penalty(self, real_images, fake_images):
        args = self.args
        eta = torch.FloatTensor(args.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(args.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda: eta = eta.cuda() 

        interpolated = eta * real_images + ((1 - eta) * fake_images)
        if self.cuda: interpolated = interpolated.cuda()
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                grad_outputs=torch.ones(prob_interpolated.size()).cuda() 
                                    if self.cuda else torch.ones(prob_interpolated.size()),
                                create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty


    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda()
        else:
            return Variable(arg)


    def run(self, data):

        args = self.args
        if not os.path.exists('./gan_imgs/' + args.dataset + '/' + args.tag):
            os.mkdir('./gan_imgs/' + args.dataset + '/' + args.tag)
        if not os.path.exists('./saved_models/' + args.dataset + '/' + args.tag):
            os.mkdir('./saved_models/' + args.dataset + '/' + args.tag)

        img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])  # (x-mean) / std
        ])
        train_dataset = dataLoader(data=data, mode='train', transform=img_transform)
        self.dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=args.lr, betas=(0.5, 0.999))

        self.train()
        

    def train(self):
        
        args = self.args
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(self.dataloader)

        one = torch.tensor(1, dtype=torch.float).type(Tensor)
        mone = one * -1

        for g_iter in range(args.gan_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                images = self.data.__next__()
                # Check for batch to have full batch_size
                if (images.size()[0] != args.batch_size):
                    continue

                z = torch.rand((args.batch_size, args.gan_z_dim, 1, 1))

                images, z = self.get_torch_variable(images), self.get_torch_variable(z)

                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = self.get_torch_variable(torch.randn(args.batch_size, args.gan_z_dim, 1, 1))

                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                gradient_penalty.backward()


                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')
            
            
            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            
            z = self.get_torch_variable(torch.randn(args.batch_size, args.gan_z_dim, 1, 1))
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean()
            self.G.zero_grad()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            print(f'Generator iteration: {g_iter}/{args.gan_iters}, g_loss: {g_loss}')
            # Saving model and sampling images every 1000th generator iterations
            
                
            if not os.path.exists('gan_imgs/'):
                os.makedirs('gan_imgs/')

            # Denormalize images and save them in grid 8x8
            z = self.get_torch_variable(torch.randn(args.batch_size, args.gan_z_dim, 1, 1))
            samples = self.G(z)
            samples = samples.mul(0.5).add(0.5)
            samples = samples.data.cpu()[:64]
            grid = torchvision.utils.make_grid(samples)

            if (g_iter) % args.save_model_interval == 0:
                self.save_model()
                torchvision.utils.save_image(grid, './gan_imgs/' + args.dataset + '/' + args.tag + '/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))

                # Testing
                #print("Real Inception score: {}".format(inception_score))
                print("Generator iter: {}".format(g_iter))

        self.save_model()

    def save_model(self):
        torch.save(self.G.state_dict(), './saved_models/generator.pkl')
        torch.save(self.D.state_dict(), './saved_models/discriminator.pkl')
        print('Models save to ./saved_models/generator.pkl & ./saved_models/discriminator.pkl ')

    
