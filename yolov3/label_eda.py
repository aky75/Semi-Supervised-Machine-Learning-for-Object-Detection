#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import cv2
import timeit


# In[ ]:


filepath = Path('/home/aditya/yolov3/final/data/kitti_labels')


# In[ ]:


exp_filepath = filepath / '000008.txt'
exp_filepath.open().read()


# In[ ]:


def parse_label_feature(s):
    pairs = s.strip().split('\n')
    pairs = [i.split(' ') for i in pairs]
    return pairs


# In[ ]:


sample_label_list = parse_label_feature(exp_filepath.open().read())
type(sample_label_list[1]),sample_label_list[1]


# In[ ]:


label_df1 = pd.DataFrame(data = sample_label_list, index = None)
label_df1.head()


# In[ ]:


def parse_label_count(s):
    pairs = s.strip().split('\n')
    obj_count = len(pairs)
    return obj_count


# In[ ]:


max_iterations = 7480

k = []
label_name = []
for x in list(filepath.glob('*.txt'))[:max_iterations]:
  k = parse_label_count(x.open().read())
  f = x.name.replace('.txt','')
  label_name += [f]*k 

print(len(label_name))


# In[ ]:


max_iterations = 7480

label_df = []
for x in list(filepath.glob('*.txt'))[:max_iterations]:
  label_df += (parse_label_feature(x.open().read()))

print(len(label_df))


# In[ ]:


label_df[0:2]


# In[ ]:


label_df1 = pd.DataFrame(data = label_df, index = label_name )
label_df1.head()


# In[ ]:


label_df1.columns = ['Image_class','truncated','occluded','alpha','l_bbox','t_bbox','r_bbox','b_bbox','h(m)','w(m)','l(m)','loc_x','loc_y','loc_z','rotation_y']
label_df1.head()


# In[ ]:


class_dist = label_df1.groupby('Image_class', as_index=False ).count()
class_dist.head()


# In[ ]:


x = class_dist.Image_class
y = class_dist.truncated


# In[ ]:


bplot = sns.barplot(x,y)
bplot.set_xticklabels(bplot.get_xticklabels(), rotation=45)
bplot.set(ylabel = "Count")


# In[ ]:


occluded_img = label_df1.groupby('occluded', as_index=False ).count()
occluded_img


# In[ ]:


label_df1[label_df1.occluded =='-1'].head()


# In[3]:


CHANNEL_NUM = 3
pixel_num = 0 # store all pixel number in the dataset
channel_sum = np.zeros(CHANNEL_NUM)
channel_sum_squared = np.zeros(CHANNEL_NUM)


# In[4]:


with open('/home/aditya/yolov3/final/pathtofile.txt') as f:
    pathlist = f.readlines()


# In[5]:


for im_path in pathlist:
    path = im_path.strip()
    im = cv2.imread(path)
    im = im/255.0
    pixel_num += (im.size/CHANNEL_NUM)
    channel_sum += np.sum(im, axis=(0, 1))
    channel_sum_squared += np.sum(np.square(im), axis=(0, 1))


# In[6]:


bgr_mean = channel_sum / pixel_num
bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))


# In[7]:


rgb_mean = list(bgr_mean)[::-1]
rgb_std = list(bgr_std)[::-1]

print(rgb_mean, rgb_std)


# In[ ]:




