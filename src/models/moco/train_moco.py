import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import copy
import cv2 as cv2
import glob
from sklearn.model_selection import train_test_split


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x




class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p



class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x

class unet_enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        self.fc1 = nn.Flatten()

    def forward(self, inputs):
        self.s1, p1 = self.e1(inputs)
        self.s2, p2 = self.e2(p1)
        self.s3, p3 = self.e3(p2)
        self.s4, p4 = self.e4(p3)
        fl = self.fc1(p4)
        return fl


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.normalize(x)
        return x
class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        self.b = conv_block(512, 1024)

        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):

        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b = self.b(p4)


        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)


        outputs = self.outputs(d4)

        return outputs

class DuplicatedCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        img1 = img.copy()
        img2 = img.copy()
        for t in self.transforms:
            #print(str(t))
            img1 = t(img1)
            img2 = t(img2)
        return img1, img2


def momentum_update(model_q, model_k, beta=0.999):
    param_k = model_k.state_dict()
    param_q = model_q.named_parameters()
    for n, q in param_q:
        if n in param_k:
            param_k[n].data.copy_(beta * param_k[n].data + (1 - beta) * q.data)
    model_k.load_state_dict(param_k)


def queue_data(data, k):
    #(data.shape, k.shape)
    return torch.cat([data, k], dim=0)


def dequeue_data(data, K=131072):
    if len(data) > K:
        return data[-K:]
    else:
        return data


def initialize_queue(model_k, device, train_loader):
    queue = torch.zeros((0, 524288), dtype=torch.float)
    queue = queue.to(device)

    for batch_idx, (data, target) in enumerate(train_loader):
        x_k = data[1]
        x_k = x_k.to(device)
        k = model_k(x_k.unsqueeze(0))
        k = k.detach()
        queue = queue_data(queue, k)
        queue = dequeue_data(queue, K=10)
        break
    return queue


def train(model_q, model_k, device, train_loader, queue, optimizer, epoch, temp=0.07):
    model_q.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        x_q = data
        x_k = data

        x_q, x_k = x_q.to(device), x_k.to(device)
        q = model_q(x_q)
        k = model_k(x_k)
        k = k.detach()

        N = data.shape[0]

        '''
        
        print(data.shape)
        print(x_q.shape)
        print(data.shape[0])
        print(q.shape, k.shape)
        
        '''
        K = queue.shape[0]
        l_pos = torch.bmm(q.view(N, 1, -1), k.view(N, -1, 1)).detach()
        l_neg = torch.mm(q.view(N, -1), queue.T.view(-1, K)).detach()

        logits = torch.cat([l_pos.view(N, 1), l_neg], dim=1).detach()

        labels = torch.zeros(N, dtype=torch.long).detach()
        labels = labels.to(device).detach()

        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(logits / temp, labels)
        loss.requires_grad = True
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(total_loss)
        momentum_update(model_q, model_k)

        queue = queue_data(queue, k)
        queue = dequeue_data(queue)
        torch.cuda.empty_cache()

    total_loss /= len(train_loader.dataset)

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss))

class TransformedDataset(Dataset):
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        image = self.tensors[index]
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.tensors)

class Dataset:
    def __init__(self, images_path, masks_path, train_val_split, shape):
        images_path += "/*.png"
        masks_path += "/*.png"
        self.images = [Image.fromarray(cv2.resize(cv2.imread(fname), shape)) for fname in
                       glob.glob(images_path)]
        self.masks = [Image.fromarray(cv2.resize(cv2.imread(fname), shape))
                      for
                      fname in glob.glob(masks_path)]
        self.train_val_split = train_val_split
        self.__split()

    def __split(self):
        self.train_images, self.val_images, self.train_masks, self.val_masks = train_test_split(self.images, self.masks,
                                                                                                test_size=self.train_val_split)

    def get_splitted(self):
        return self.train_images, \
               self.val_images, \
               self.train_masks, \
               self.val_masks


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MoCo example: MNIST')
    parser.add_argument('images', '--images-path', type=str,
                        help='path to images')
    parser.add_argument('masks', '--masks-path', type=str,
                        help='path to binary masks')
    parser.add_argument('out', '--out-dir', type=str,
                        help='out dir to save results')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    images_path = args.images
    masks_path = args.masks

    X_train, X_val, Y_train, Y_val = Dataset(images_path, masks_path,
                                             0.1, (512, 512)).get_splitted()

    batchsize = 8
    epochs = 10
    out_dir = args.out

    use_cuda = torch.cuda.is_available()
    device = "cpu"

    kwargs = {'num_workers': 0, 'pin_memory': True}

    transform = DuplicatedCompose([
        transforms.ToTensor(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(512, scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=2),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = TransformedDataset(X_train, transform=transform)
    val_dataset = TransformedDataset(Y_train, transform=transform)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True, **kwargs)

    model_q = unet_enc().to(device)
    model_k = copy.deepcopy(model_q)
    optimizer = optim.SGD(model_q.parameters(), lr=0.01, weight_decay=0.0001)

    queue = initialize_queue(model_k, device, train_loader)

    for epoch in range(1, epochs + 1):
        train(model_q, model_k, device, train_loader, queue, optimizer, epoch)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model_q.state_dict(), os.path.join(out_dir, 'model.pth'))