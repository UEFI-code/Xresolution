import torch
import torch.nn as nn
import torch.nn.functional as F
import model
import cv2

device = 'cpu'

model = model.myModel().to(device)
model.eval()
model.load_state_dict(torch.load('model.pth', map_location=device))

def infer(img):
    img = img.transpose((2, 0, 1))
    img = torch.tensor(img).float().unsqueeze(0).to(device) / 255
    out = model(img)
    return out[0].detach().numpy().transpose((1, 2, 0))

def testCamera(devID):
    cam = cv2.VideoCapture(devID)
    while True:
        ret, img = cam.read()
        out = infer(img)
        cv2.imshow('image', out)
        cv2.waitKey(1)

def testImg(path):
    img = cv2.imread(path)
    out = infer(img)
    cv2.imshow('image', out)
    cv2.waitKey(0)

#testImg('test.png')
testCamera(0)
