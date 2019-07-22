#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 15:10:40 2019

@author: liuhongbing
"""

from collections import Counter
import numpy as np
import tensorflow as tf


"""
诗歌转为 向量
"""
def getPoetryList(root):
    
    poetrylist = []
    with open(root+"/poetry.txt", 'r') as f:
        for line in f.readlines():
            content = line.strip().split(':')[1]
            if u'_' in content or u'(' in content or u'（' in content or u'《' in content or u'[' in content:
                continue
            if '2' in content or '3' in content:
                continue
            if len(content)<5 or  len(content) > 79: 
                continue
            
            content = u'[' + content + u']'
            poetrylist.append(content)
        
        
    allword = []
    for poetry in poetrylist:
        allword += [word for word in poetry]
    
    
    counter = Counter(allword)  
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  
    words, _ = zip(*count_pairs)  
    
    words = words[:len(words)] + ('UNK',)
    word_num_map = dict(zip(words, range(len(words))))    
    
    poetry_vector = []
    for poetry in poetrylist:
        vector = [word_num_map.get(word, len(words)) for word in poetry]
        poetry_vector.append(vector)
    return poetry_vector,word_num_map,words


class DataSet(object):
    
    def __init__(self,data_size,poetry_vector,word_num_map):
        self._data_size = data_size
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._data_index = np.arange(data_size)
        self.poetry_vector = poetry_vector
        self.word_num_map = word_num_map
        

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        if start + batch_size > self._data_size:
            np.random.shuffle(self._data_index)
            self._epochs_completed = self._epochs_completed + 1
            self._index_in_epoch = batch_size
            full_batch_features ,full_batch_labels = self.data_batch(0,batch_size)
            return full_batch_features ,full_batch_labels 
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            full_batch_features ,full_batch_labels = self.data_batch(start,end)
            if self._index_in_epoch == self._data_size:
                self._index_in_epoch = 0
                self._epochs_completed = self._epochs_completed + 1
                np.random.shuffle(self._data_index)
            return full_batch_features,full_batch_labels


    def data_batch(self,start,end):
        batches = []
        for i in range(start,end):
            batches.append(self.poetry_vector[self._data_index[i]])

        length = max(map(len,batches))

        xdata = np.full((end - start,length), self.word_num_map['UNK'], np.int32)  
        for row in range(end - start):  
            xdata[row,:len(batches[row])] = batches[row]  
        ydata = np.copy(xdata)  
        ydata[:,:-1] = xdata[:,1:]  
        return xdata,ydata



    
    
def load_model(sess, saver,ckpt_path):
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
        print ('resume from', latest_ckpt)
        saver.restore(sess, latest_ckpt)
        return int(latest_ckpt[latest_ckpt.rindex('-') + 1:])
    else:
        print ('building model from scratch')
        sess.run(tf.global_variables_initializer())
        return -1   
    
    
    
    
    
    
    
    
    