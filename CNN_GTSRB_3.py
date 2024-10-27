import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from torch.utils.data import Dataset,DataLoader
import torchvision
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2 as cv
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.optim import lr_scheduler
import os

data_dir = "/home/mohsen/Documents/dataset/gtsrb"
train_path = "/home/mohsen/Documents/dataset/gtsrb/GTSRB/Training"
test_path = "/home/mohsen/Documents/dataset/gtsrb/GTSRB/Final_Test"
train_df = pd.read_csv('/home/mohsen/Documents/dataset/gtsrb/GTSRB/Training/Train.csv')
test_df = pd.read_csv('/home/mohsen/Documents/dataset/gtsrb/Te  st.csv')
transforms = transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),transforms.Resize((32,32))])

batch_size = 4
classes = {0: 'Speed limit (20km/h)',
1: 'Speed limit (30lom/h)',
2: 'Speed limit (50km/h)',
3: 'Speed limit (60km/h)',
4:'Speed limit (70km/h)',
5:'Speed limit (Bokm/h)',
6: 'End of speed limit (80km/h)',
7:'Speed limit (100km/h)' ,
8: 'Speed limit (120km/h)',
9: 'No passing' ,
10: "No passing veh over 3.5 tons",
11: 'Right-o-way at intersection',
12: 'Priority road' ,
13: 'Yield',
14: 'Stop',
15: 'No vehicles',
16: 'Veh > 3.5 tons prohibited',
17: 'No entry',
18: 'General caution',
19: 'Dangerous curve left',
20: "Dangerous curve right",
21: 'Double curve',
22: 'Bumpy road',
23: 'Slippery road',
24: "Road narrows on the right",
25: "Road work",
26: 'Traffic signals',
27: 'Pedestrians',
28: 'Children crossing',
29: 'Bicycles crossing',
30: "Beware of ice/snow",
31: "Wild animals crossing",
32: 'End speed + passing limits',
33: "Turn right ahead",
34: "Turn left ahead",
35: 'Ahead only',
36: 'Go straight or right',
37: 'Go straight or left',
38: 'Keep right',
39: 'Keep left',
40: "Roundabout mandatory",
41: 'End of no passing',
42: 'End no passing veh> 3.5 tons' }


class GTSRB(Dataset):
    def __init__(self,df,root_dir,transform = None):
      self.df = df
      self.root_dir = root_dir
      self.transform = transform


    def __len__(self):
      return len(self.df)
   
   
    def __getitem__(self, index):
      image_path= os.path.join(self.root_dir,self.df.iloc[index,7])
      image = Image.open(image_path)
      y_class = torch.tensor(self.df.iloc[index,6])

      if self.transform:
         image = self.transform(image)
         return (image,y_class)
      



        
train_dataset = GTSRB(train_df,train_path,transform=transforms)
test_dataset = GTSRB(test_df,test_path,transform=transforms)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size=batch_size,shuffle=False)

dataloaders = {'training':train_loader,'testing':test_loader}
dataset_sizes = {'training':len(train_loader.dataset),'testing':len(test_loader.dataset)}
print(dataset_sizes)





# data_iter = iter(train_loader)
# image , labels = next(data_iter)

# # print(images.shape)
# # print(labels.shape)


# def imshow(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# imshow(torchvision.utils.make_grid(image))




class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(8*8*32,128)
        self.fc2 = nn.Linear(128,len(classes))


    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,32*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x






model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
num_epochs = 1

for epoch in range(num_epochs):
    for i ,(images , labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{5}], Step [{i+1}/ {(len(train_loader))}], Loss: {loss.item():.4f}')




with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(43)]
    n_class_samples = [0 for i in range(43)]
    for images, labels in test_loader:
        images = images
        labels = labels
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
          try:
            if (labels[i] == predicted[i]):
              n_class_correct[labels[i]] += 1
            n_class_samples[labels[i]] += 1
          except IndexError:
            continue

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(43):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')




