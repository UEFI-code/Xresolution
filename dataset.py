import torch
import cv2
import numpy as np
import os
import tqdm

class ImgDataset():
    def __init__(self, root_dir, bigMemory=False):
        self.root_dir = root_dir
        self.img_paths = os.listdir(root_dir)
        self.length = len(self.img_paths)
        self.enumIndex = 0
        self.bigMemory = bigMemory
        if bigMemory:
            print("Wow big RAM! Loading images into memory...")
            self.imgMemory = []
            for i in tqdm.tqdm(range(self.length)):
                img_path = os.path.join(root_dir, self.img_paths[i])
                image = cv2.imread(img_path)
                self.imgMemory.append(image)

    def __getitem__(self, idx, resolution=(256, 256)):
        if self.bigMemory:
            image = self.imgMemory[idx]
        else:
            img_path = os.path.join(self.root_dir, self.img_paths[idx])
            image = cv2.imread(img_path)
        image = cv2.resize(image, resolution)
        image = image.transpose((2, 0, 1))
        return image
    
    def makeBatch(self, batch_size, resolution=(256, 256)):
        batch = []
        for i in range(batch_size):
            batch.append(self.__getitem__((self.enumIndex + i) % self.length, resolution))
            #self.enumIndex += 1
        #self.enumIndex %= self.length # Ensure that the index is always in range
        return torch.tensor(batch).float()
    
    def step(self, step_size):
        self.enumIndex += step_size
        self.enumIndex %= self.length

class VideoDataset():
    def __init__(self, root_dir, bigMemory=False):
        self.root_dir = root_dir
        self.video_paths = os.listdir(root_dir)
        self.enumIndex = 0
        self.bigMemory = bigMemory
        if bigMemory:
            print("Wow big RAM! Loading videos into memory...")
            self.videoMemory = []
            for i in self.video_paths:
                if not i.endswith('.mp4'):
                    continue
                video_path = os.path.join(root_dir, i)
                video = cv2.VideoCapture(video_path)
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    self.videoMemory.append(frame)
                video.release()
            self.length = len(self.videoMemory)
            self.totalFrames = self.length
        else:
            self.length = []
            self.videoHandles = []
            for i in self.video_paths:
                if not i.endswith('.mp4'):
                    continue
                video_path = os.path.join(root_dir, i)
                video = cv2.VideoCapture(video_path)
                self.length.append(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
                self.videoHandles.append(video)
                print("Video {} has {} frames".format(i, self.length[-1]))
            self.totalFrames = sum(self.length)
    
    def __getitem__(self, idx, resolution=(256, 256)):
        if self.bigMemory:
            image = self.videoMemory[idx]
        else:
            internal_file_index = 0
            s = self.length[0]
            while idx >= s:
                internal_file_index += 1
                s += self.length[internal_file_index]
            s -= self.length[internal_file_index]
            #print('Selected video: {}'.format(self.video_paths[internal_file_index]))
            video = self.videoHandles[internal_file_index]
            video.set(cv2.CAP_PROP_POS_FRAMES, idx - s)
            ret, image = video.read()
        image = cv2.resize(image, resolution)
        image = image.transpose((2, 0, 1))
        return image
    
    def makeBatch(self, batch_size, resolution=(256, 256)):
        batch = []
        for i in range(batch_size):
            batch.append(self.__getitem__((self.enumIndex + i) % self.totalFrames, resolution))
            #self.enumIndex += 1
        #self.enumIndex %= totalFrames # Ensure that the index is always in range
        return torch.tensor(batch).float()
    
    def step(self, step_size):
        self.enumIndex += step_size
        self.enumIndex %= self.totalFrames
