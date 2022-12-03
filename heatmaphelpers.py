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
import matplotlib.cm
import cv2


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


def apply_custom_colormap(image_gray, cmap=plt.get_cmap('seismic')):

    assert image_gray.dtype == np.uint8, 'must be np.uint8 image'
    if image_gray.ndim == 3: image_gray = image_gray.squeeze(-1)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256))[:,0:3]    # color range RGBA => RGB
    color_range = (color_range*255.0).astype(np.uint8)         # [0,1] => [0,255]

    color_range = np.squeeze(np.dstack([color_range[:,0], color_range[:,1], color_range[:,2]]), 0)  # RGB => RGB
    # color_range = np.squeeze(np.dstack([color_range[:,2], color_range[:,1], color_range[:,0]]), 0)  # RGB => BGR

    # Apply colormap for each channel individually
    channels = [cv2.LUT(image_gray, color_range[:,i]) for i in range(3)]
    return np.dstack(channels)





def hm_to_rgb(R, X = None, sigma = 2, cmap = 'seismic', normalize = True):
    '''
    Takes as input an intensity array and produces a rgb image for the represented heatmap.
    optionally draws the outline of another input on top of it.
    Parameters
    ----------
    R : numpy.ndarray
        the heatmap to be visualized, shaped [M x N]
    X : numpy.ndarray
        optional. some input, usually the data point for which the heatmap R is for, which shall serve
        as a template for a black outline to be drawn on top of the image
        shaped [M x N]
    scaling: int
        factor, on how to enlarge the heatmap (to control resolution and as a inverse way to control outline thickness)
        after reshaping it using shape.
    shape: tuple or list, length = 2
        optional. if not given, X is reshaped to be square.
    sigma : double
        optional. sigma-parameter for the canny algorithm used for edge detection. the found edges are drawn as outlines.
    cmap : str
        optional. color map of choice
    normalize : bool
        optional. whether to normalize the heatmap to [-1 1] prior to colorization or not.
    Returns
    -------
    rgbimg : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3] , where H*W == M*N
    '''

    if cmap in custom_maps:
        rgb =  custom_maps[cmap](R)
    else:
        if normalize:
            R = R / np.amax(np.abs(R)) # normalize to [-1,1] wrt to max relevance magnitude
            R = (R + 1.)/2. # shift/normalize to [0,1] for color mapping

        #create color map object from name string
        cmap = eval('matplotlib.cm.{}'.format(cmap))

        # apply colormap
        # print(cmap(R.flatten()).shape)
        rgb = cmap(R.flatten())[...,0:3].reshape([R.shape[0],R.shape[1],3])
    #rgb = repaint_corner_pixels(rgb, scaling) #obsolete due to directly calling the color map with [0,1]-normalized inputs

    if not X is None: #compute the outline of the input
        # X = enlarge_image(vec2im(X,shape), scaling)
        X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        xdims = X.shape
        Rdims = R.shape

        if not np.all(xdims == Rdims):
            print('transformed heatmap and data dimension mismatch. data dimensions differ?')
            print('R.shape = ',Rdims, 'X.shape = ', xdims)
            print('skipping drawing of outline\n')
        else:
            edges = canny(X, sigma=sigma)
            edges = np.invert(np.dstack([edges]*3))*1.0
            rgb *= edges # set outline pixels to black color

    return rgb





# ################## #
# custom color maps: #
# ################## #

def gregoire_gray_red(R):
    basegray = 0.8 #floating point gray

    maxabs = np.max(R)
    RGB = np.ones([R.shape[0], R.shape[1],3]) * basegray #uniform gray image.

    tvals = np.maximum(np.minimum(R/maxabs,1.0),-1.0)
    negatives = R < 0

    RGB[negatives,0] += tvals[negatives]*basegray
    RGB[negatives,1] += tvals[negatives]*basegray
    RGB[negatives,2] += -tvals[negatives]*(1-basegray)

    positives = R>=0
    RGB[positives,0] += tvals[positives]*(1-basegray)
    RGB[positives,1] += -tvals[positives]*basegray
    RGB[positives,2] += -tvals[positives]*basegray

    return RGB


def gregoire_black_green(R):
    maxabs = np.max(R)
    RGB = np.zeros([R.shape[0], R.shape[1],3])

    negatives = R<0
    RGB[negatives,2] = -R[negatives]/maxabs

    positives = R>=0
    RGB[positives,1] = R[positives]/maxabs

    return RGB


def gregoire_black_firered(R):
    R = R / np.max(np.abs(R))
    x = R

    hrp  = np.clip(x-0.00,0,0.25)/0.25
    hgp = np.clip(x-0.25,0,0.25)/0.25
    hbp = np.clip(x-0.50,0,0.50)/0.50

    hbn = np.clip(-x-0.00,0,0.25)/0.25
    hgn = np.clip(-x-0.25,0,0.25)/0.25
    hrn = np.clip(-x-0.50,0,0.50)/0.50

    return np.concatenate([(hrp+hrn)[...,None],(hgp+hgn)[...,None],(hbp+hbn)[...,None]],axis = 2)


def gregoire_gray_red2(R):
    v = np.var(R)
    R[R > 10*v] = 0
    R[R<0] = 0
    R = R / np.max(R)
    #(this is copypasta)
    x=R

    # positive relevance
    hrp = 0.9 - np.clip(x-0.3,0,0.7)/0.7*0.5
    hgp = 0.9 - np.clip(x-0.0,0,0.3)/0.3*0.5 - np.clip(x-0.3,0,0.7)/0.7*0.4
    hbp = 0.9 - np.clip(x-0.0,0,0.3)/0.3*0.5 - np.clip(x-0.3,0,0.7)/0.7*0.4

    # negative relevance
    hrn = 0.9 - np.clip(-x-0.0,0,0.3)/0.3*0.5 - np.clip(-x-0.3,0,0.7)/0.7*0.4
    hgn = 0.9 - np.clip(-x-0.0,0,0.3)/0.3*0.5 - np.clip(-x-0.3,0,0.7)/0.7*0.4
    hbn = 0.9 - np.clip(-x-0.3,0,0.7)/0.7*0.5

    hr = hrp*(x>=0)+hrn*(x<0)
    hg = hgp*(x>=0)+hgn*(x<0)
    hb = hbp*(x>=0)+hbn*(x<0)


    return np.concatenate([hr[...,None],hg[...,None],hb[...,None]],axis=2)



def alex_black_yellow(R):

    maxabs = np.max(R)
    RGB = np.zeros([R.shape[0], R.shape[1],3])

    negatives = R<0
    RGB[negatives,2] = -R[negatives]/maxabs

    positives = R>=0
    RGB[positives,0] = R[positives]/maxabs
    RGB[positives,1] = R[positives]/maxabs

    return RGB


#list of supported color map names. the maps need to be implemented ABOVE this line because of PYTHON
custom_maps = {'gray-red':gregoire_gray_red,\
'gray-red2':gregoire_gray_red2,\
'black-green':gregoire_black_green,\
'black-firered':gregoire_black_firered,\
'blue-black-yellow':alex_black_yellow}