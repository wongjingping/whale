# -*- coding: utf-8 -*-
"""
crop pics using the json annotations provided by anlthms

@author: jingpingw
"""

from time import time
import json
import math
import numpy as np
import cv2


path_img = '/Users/jingpingw/Documents/whale/imgs/'

with open(path_img+'data/p1.json') as j1:
    p1 = json.load(j1)
with open(path_img+'data/p2.json') as j2:
    p2 = json.load(j2)
assert(len(p1)==len(p2))

ext = 1.5

t_start = time()
for i in range(len(p1)):
    # read in image and annotation
    assert(p1[i]['filename'] == p2[i]['filename'])
    f_ = p1[i]['filename']
    ima = cv2.imread(path_img+'raw/'+f_) # BGR format!
    x1, y1 = int(p1[i]['annotations'][0]['x']), int(p1[i]['annotations'][0]['y'])
    x2, y2 = int(p2[i]['annotations'][0]['x']), int(p2[i]['annotations'][0]['y'])
    # pre-crop image with boundary checks for padding
    rows,cols = ima.shape[0:2]
    dx, dy = x1-x2, y1-y2
    dist = int(math.hypot(dx,dy) * ext) 
    if x2-dist < 0:
        im_xmin,imc_xmin = 0,dist-x2
    else:
        im_xmin,imc_xmin = x2-dist,0
    if x2+dist > cols:
        im_xmax,imc_xmax = cols,cols-x2+dist
    else:
        im_xmax,imc_xmax = x2+dist,2*dist
    if y2-dist < 0:
        im_ymin,imc_ymin = 0,dist-y2
    else:
        im_ymin,imc_ymin = y2-dist,0
    if y2+dist > rows:
        im_ymax,imc_ymax = rows,int(rows-y2+dist)
    else:
        im_ymax,imc_ymax = int(y2+dist),2*dist
    imc = ima[im_ymin:im_ymax,im_xmin:im_xmax,:]
    imcp = np.zeros((2*dist,2*dist,3),dtype=np.uint8)
    imcp[imc_ymin:imc_ymax,imc_xmin:imc_xmax,:] = imc
    # rotate image about (dist,dist)
    angle = math.atan2(dy,dx) / math.pi * 180 + 90
    R = cv2.getRotationMatrix2D((dist,dist),angle,1)
    imcpr = cv2.warpAffine(imcp,R,(2*dist,2*dist))
    # crop with extra 10% for bottom
    imcprc = imcpr[int(dist*0.2):int(dist*1.2),int(dist*0.5):int(dist*1.5),:]
    # resize for lower memory footprint
    imcprcs = cv2.resize(imcprc, (100,130))
    cv2.imwrite(path_img+'crop/'+f_,imcprcs)

print('Cropping %i images took %.1fs' % (len(p1),time()-t_start))


