import torch
import torch.nn as nn
import torchvision
import numpy as np


class DataLoader():
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
        print(self.mode, self.num_dataset)

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
    

class MyNet(nn.Module):

    def __init__(self, n_class=10):
        super(MyNet, self).__init__()
        # input: batch_size * 3 * 32 * 32
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # batch_size * 32 * 32 * 32
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # batch_size * 64 * 32 * 32
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True), 

            nn.MaxPool2d(kernel_size=2, stride=2),

            # batch_size * 128 * 16 * 16
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*8*8, 100),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(100, n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.network(x)
        x = x.view(-1, 128*8*8)
        output = self.classifier(x)
        return output

    def run(self, data, n_epoch=10, batch_size=256, lr=0.0001):
        
        img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))  # (x-mean) / std
        ])
        train_dataset = DataLoader(data=data, mode='train', transform=img_transform)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = DataLoader(data=data, mode='test', transform=img_transform)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # model = torchvision.models.resnet18(pretrained=False, num_classes=1).cuda()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        Loss = nn.BCELoss().cuda()

        train(n_epoch, self.train_loader, self, optimizer, Loss)
        test_loss, correct, total = test(self.test_loader, self, Loss)
        print('\nTesting: Mean_loss:{:.4f}    Acc:{}/{} ({:.2f}%)\n'.format(
            test_loss, correct, total, correct/total))

def train(n_epoch, train_loader, model, optimizer, Loss):

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    for epoch in range(n_epoch):  
        for batch_idx, (img, target) in enumerate(train_loader):
            img = img.type(Tensor)
            target = target.type(Tensor)

            output = model(img)
            optimizer.zero_grad()
            loss = Loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                _, correct, total = test(train_loader, model, Loss)
                print('Epoch: [{}/{}]    Batch:{}/{}   Train_acc:{}    Loss:{:.6f}'.format(
                    epoch, n_epoch, batch_idx, len(train_loader), correct/total, loss.item()/img.shape[0]))
    torch.save(model.state_dict(), 'saved_models/model'+str(n_epoch)+".pth")



def test(test_loader, model, Loss):
    test_loss = 0
    correct = 0
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    total = 0
    for img, target in test_loader:
        img = img.type(Tensor)
        output = model(img)
        test_loss += Loss(output, target.type(Tensor)).item()

        # get the index of the max
        pred = torch.argmax(output.data.cpu(), axis=1)
        target = torch.argmax(target, axis=1)
        correct += torch.sum(pred==target)
        total += len(img)
    test_loss /= total
    return test_loss, correct, total



def main():
    data = np.load("data/vec/data_1.npz") 

    model = MyNet().cuda()
    model.run(data)

    


# model = torchvision.models.resnet18(pretrained=True)
# print(model)

if __name__ == "__main__":
    main()