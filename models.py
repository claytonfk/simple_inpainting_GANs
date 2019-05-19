# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:46:30 2019

@author: souzac1, azizazh
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    # Defines the generator architecture
    def __init__(self):
        super(Generator, self).__init__()

        self.block1 = torch.nn.Sequential()
        self.block1.add_module("conv_1", torch.nn.Conv2d(3, 64, kernel_size=4, padding=2))
        self.block1.add_module("relu_1", torch.nn.ReLU())
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.block2 = torch.nn.Sequential()
        self.block2.add_module("conv_3", torch.nn.Conv2d(64, 128, kernel_size=4, padding=2))
        self.block2.add_module("relu_3", torch.nn.ReLU())
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.block3 = torch.nn.Sequential()
        self.block3.add_module("conv_5", torch.nn.Conv2d(128, 256, kernel_size=4, padding=2))
        self.block3.add_module("relu_5", torch.nn.ReLU())
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.block4 = torch.nn.Sequential()
        self.block4.add_module("conv_7", torch.nn.Conv2d(256, 512, kernel_size=4, padding=2))
        self.block4.add_module("relu_7", torch.nn.ReLU())
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
               
        self.block5 = torch.nn.Sequential()
        self.block5.add_module("conv_9", torch.nn.Conv2d(512, 512, kernel_size=4, padding=2))
        self.block5.add_module("relu_9", torch.nn.ReLU())
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.block5e = torch.nn.Sequential()
        self.block5e.add_module("conv_9e", torch.nn.Conv2d(512, 512, kernel_size=4, padding=2))
        self.block5e.add_module("relu_9e", torch.nn.ReLU())
        self.pool5e = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        #self.fc1 = nn.Linear(8192, 8192)  
        #self.fc2 = nn.Linear(8192, 8192)
        
        self.unpool1e = nn.MaxUnpool2d(2, stride=2)
        self.block6e = torch.nn.Sequential()
        self.block6e.add_module("deconv_1e", torch.nn.Conv2d(512, 512, kernel_size=4, padding=1))
        self.block6e.add_module("relu_9e", torch.nn.ReLU())
        
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.block6 = torch.nn.Sequential()
        self.block6.add_module("deconv_1", torch.nn.Conv2d(512, 512, kernel_size=4, padding=1))
        self.block6.add_module("relu_9", torch.nn.ReLU())
       
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.block7 = torch.nn.Sequential()
        self.block7.add_module("deconv_2", torch.nn.Conv2d(512, 256, kernel_size=4, padding=1))
        self.block7.add_module("relu_10", torch.nn.ReLU())

        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.block8 = torch.nn.Sequential()
        self.block8.add_module("deconv_4", torch.nn.Conv2d(256, 128, kernel_size=4, padding=1))
        self.block8.add_module("relu_12", torch.nn.ReLU())

        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.block9 = torch.nn.Sequential()
        self.block9.add_module("deconv_6", torch.nn.Conv2d(128, 64, kernel_size=4, padding=1))
        self.block9.add_module("relu_14", torch.nn.ReLU())
        
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.block10 = torch.nn.Sequential()
        self.block10.add_module("deconv_8", torch.nn.Conv2d(64, 3, kernel_size=4, padding=1))
        self.block10.add_module("tanh", torch.nn.Tanh())

        self.block1 = nn.DataParallel(self.block1)
        self.block2 = nn.DataParallel(self.block2)
        self.block3 = nn.DataParallel(self.block3)
        self.block4 = nn.DataParallel(self.block4)
        self.block5 = nn.DataParallel(self.block5)
        self.block6 = nn.DataParallel(self.block6)
        self.block7 = nn.DataParallel(self.block7)
        self.block8 = nn.DataParallel(self.block8)
        self.block9 = nn.DataParallel(self.block9)
        self.block10 = nn.DataParallel(self.block10)
        
        self.pool1 = nn.DataParallel(self.pool1)
        self.pool2 = nn.DataParallel(self.pool2)
        self.pool3 = nn.DataParallel(self.pool3)
        self.pool4 = nn.DataParallel(self.pool4)
        self.pool5 = nn.DataParallel(self.pool5)
        
        self.pool5e = nn.DataParallel(self.pool5e)
        
        self.unpool1e = nn.DataParallel(self.unpool1e)
        
        self.unpool1 = nn.DataParallel(self.unpool1)
        self.unpool2 = nn.DataParallel(self.unpool2)
        self.unpool3 = nn.DataParallel(self.unpool3)
        self.unpool4 = nn.DataParallel(self.unpool4)
        self.unpool5 = nn.DataParallel(self.unpool5)

    def forward(self, x, verbose = False):
        # ENCODER
        if verbose: print(x.shape)
        x = self.block1.forward(x)
        size1 = x.size()
        x, indices1 = self.pool1(x)
        if verbose: print(x.shape)
        x = self.block2.forward(x)
        size2 = x.size()
        x, indices2 = self.pool2(x)
        if verbose: print(x.shape)
        x = self.block3.forward(x)
        size3 = x.size()
        x, indices3 = self.pool3(x)
        if verbose: print(x.shape)
        x = self.block4.forward(x)
        size4 = x.size()
        x, indices4 = self.pool4(x)
        if verbose: print(x.shape)
        x = self.block5.forward(x)
        size5 = x.size()
        x, indices5 = self.pool5(x)
        if verbose: print(x.shape)
        
        x = self.block5e.forward(x)
        size5e = x.size()
        x, indices5e = self.pool5e(x)
        if verbose: print(x.shape)

        features = x
        
        # DECODER
        
        x = self.unpool1e(x, indices5e, output_size=size5e)
        x = self.block6e.forward(x)
        if verbose: print(x.shape)
        
        x = self.unpool1(x, indices5, output_size=size5)
        x = self.block6.forward(x)
        if verbose: print(x.shape)
        x = self.unpool2(x, indices4, output_size=size4)
        x = self.block7.forward(x)
        if verbose: print(x.shape)
        x = self.unpool3(x, indices3, output_size=size3)
        x = self.block8.forward(x)
        if verbose: print(x.shape)
        x = self.unpool4(x, indices2, output_size=size2)
        x = self.block9.forward(x)
        if verbose: print(x.shape)
        x = self.unpool5(x, indices1, output_size=size1)
        x = self.block10.forward(x)
        if verbose: print(x.shape)
        
        return x, features
    
    
class Discriminator(nn.Module):
    # Defines the discriminator architecture
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        
        self.conv1 = torch.nn.Sequential()
        self.conv1.add_module("conv_1", torch.nn.Conv2d(3, 8, kernel_size=4, padding=2))
        self.conv1.add_module("relu_1", torch.nn.ReLU())
        self.conv1.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv2 = torch.nn.Sequential()
        self.conv2.add_module("conv_2", torch.nn.Conv2d(8, 16, kernel_size=4, padding=2))
        self.conv2.add_module("relu_2", torch.nn.ReLU())
        self.conv2.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv3 = torch.nn.Sequential()
        self.conv3.add_module("conv_3", torch.nn.Conv2d(16, 16, kernel_size=4, padding=2))
        self.conv3.add_module("relu_3", torch.nn.ReLU())
        self.conv3.add_module("maxpool_3", torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv4 = torch.nn.Sequential()
        self.conv4.add_module("conv_4", torch.nn.Conv2d(16, 16, kernel_size=4, padding=2))
        self.conv4.add_module("relu_4", torch.nn.ReLU())
        self.conv4.add_module("maxpool_4", torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc1 = nn.Linear(int(image_size[0]*image_size[1]/16), 256)
        self.relu = torch.nn.ReLU()
        self.fc2 = nn.Linear(256,1)
        self.sigmoid = nn.Sigmoid()
        
        self.conv1 = nn.DataParallel(self.conv1)
        self.conv2 = nn.DataParallel(self.conv2)
        self.conv3 = nn.DataParallel(self.conv3)
        self.conv4 = nn.DataParallel(self.conv4)
        self.fc1   = nn.DataParallel(self.fc1)
        self.fc2   = nn.DataParallel(self.fc2)
        self.relu  = nn.DataParallel(self.relu)
        self.sigmoid = nn.DataParallel(self.sigmoid)
        
    def forward(self, x, verbose=False):
        if verbose: print(x.shape)
        x = self.conv1.forward(x)
        if verbose: print(x.shape)
        x = self.conv2.forward(x)
        if verbose: print(x.shape)
        x = self.conv3.forward(x)
        if verbose: print(x.shape)
        x = self.conv4.forward(x)
        if verbose: print(x.shape)

        _,a,b,c = x.shape
        x = x.view(-1, a*b*c)
        
        x = self.relu(self.fc1.forward(x))
        x = self.sigmoid.forward(self.fc2.forward(x))
        return x