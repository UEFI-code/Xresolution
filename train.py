import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset
import model
import torch.optim as optim
import tqdm

# Create the model
myModel = model.myModel()
# Create the optimizer
optimizer = optim.Adam(myModel.parameters(), lr=0.001)
# Create the loss function
loss_fn = nn.MSELoss()
# Create the dataset
myDataset = dataset.ImgDataset('Download/images')
# Train the model

def train(batchsize = 16, epoch = 10, device = 'cpu', show = False):
    myModel.to(device)
    for i in range(epoch):
        for j in tqdm.tqdm(range(len(myDataset) // batchsize)):
            batchS = myDataset.makeBatch(batchsize, resolution=(256, 256)).to(device) / 255
            batchT = myDataset.makeBatch(batchsize, resolution=(512, 512)).to(device) / 255
            y = myModel(batchS)
            loss = loss_fn(y, batchT)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            myDataset.step(batchsize)
            if show and j % 100 == 0:
                showData = y[0].detach().cpu().numpy() * 255
                cv2.imshow('image', showData)
                cv2.waitKey(0)
            print('Batch: ' + str(j) + ' Loss: ' + str(loss.item()))
        
        print("Epoch: " + str(i) + " Loss: " + str(loss.item()))

train(batchsize = 16, epoch = 10, device = 'cuda', show = True)