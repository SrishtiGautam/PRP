#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:43:52 2020

@author: srishtigautam

Code built upon LRP code from https://github.com/AlexBinder/LRP_Pytorch_Resnets_Densenet
"""


import numpy as np
import matplotlib.pyplot as plt
from helpers import makedir
import torch
try:
    from skimage.feature import canny
except:
    from skimage.filter import canny



def imshow_im(hm,imgtensor,q=100,folder="folder",name="name"):

    def invert_normalize(ten, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
      s=torch.tensor(np.asarray(std,dtype=np.float32)).unsqueeze(1).unsqueeze(2)
      m=torch.tensor(np.asarray(mean,dtype=np.float32)).unsqueeze(1).unsqueeze(2)

      res=ten*s+m
      return res

    def showimgfromtensor(inpdata):

      ts=invert_normalize(inpdata)
      a=ts.data.squeeze(0).numpy()
      saveimg=(a*255.0).astype(np.uint8)

    hm = hm.squeeze().sum(dim=0).detach().numpy()
    clim = np.percentile(np.abs(hm), q)
    hm = hm / clim

    return hm