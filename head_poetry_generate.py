#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 14:51:15 2019

@author: liuhongbing
"""

import numpy as np  
import tensorflow as tf
from data_pregross import getPoetryList,DataSet,load_model

class lstm_poerty_train():
    
    def __init__(self):
        self.batch_size = 1
        self.rnn_size = 100
        self.embedding_size = 300
        self.num_layers = 2
        self.model = 'lstm'
        self.word_size = 6109
    # 定义RNN  
    def neural_network(self, xs):  
        if self.model == 'rnn':  
            cell_fun = tf.contrib.rnn.BasicRNNCell
        elif self.model == 'gru':  
            cell_fun = tf.contrib.rnn.GRUCell
        elif self.model == 'lstm':  
            cell_fun = tf.contrib.rnn.BasicLSTMCell
       
        cell1 = cell_fun(self.rnn_size, state_is_tuple=True) 
        cell2 = cell_fun(self.rnn_size, state_is_tuple=True) 

        cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2], state_is_tuple=True)
       
        initial_state = cell.zero_state(self.batch_size, tf.float32)  
        with tf.variable_scope('rnnlm'):  
            softmax_w = tf.get_variable("softmax_w", [self.rnn_size, self.word_size])  
            softmax_b = tf.get_variable("softmax_b", [self.word_size])  
            
            with tf.device("/cpu:0"):  
                embedding = tf.get_variable("embedding", [self.word_size, self.embedding_size])  
                inputs = tf.nn.embedding_lookup(embedding, xs)  
                
        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')  
        
        output = tf.reshape(outputs,[-1, self.rnn_size])  
        logits = tf.matmul(output, softmax_w) + softmax_b  
        probs = tf.nn.softmax(logits)  
        return logits, last_state, probs, cell, initial_state 
    
    
    def gen_head_poetry(self, heads, root, type):
        
        
        poetry_vector,word_num_map,words = getPoetryList(root)
        
        if type != 5 and type != 7:
            print('The second para has to be 5 or 7!')
            return
        def to_word(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            sample = int(np.searchsorted(t, np.random.rand(1)*s))
            return words[sample]
        
        input_data = tf.placeholder(tf.int32, [self.batch_size, None])  

        _, last_state, probs, cell, initial_state = self.neural_network(input_data)
        
        Session_config = tf.ConfigProto(allow_soft_placement = True)
        Session_config.gpu_options.allow_growth=True
    
        with tf.Session(config=Session_config) as sess:
#            with tf.device('/GPU:0'):
                
            sess.run(tf.global_variables_initializer())#tf.initialize_all_variables()
            saver = tf.train.Saver(tf.all_variables())
            saver.restore(sess, root + '/model/poetry.module-1')
            poem = ''
            for head in  heads:
                flag = True
                while flag:

                    state_ = sess.run(cell.zero_state(1, tf.float32))

                    x = np.array([list(map(word_num_map.get, u'['))])
                    [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})

                    sentence = head

                    x = np.zeros((1,1))
                    x[0,0] = word_num_map[sentence]
                    [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
                    word = to_word(probs_)
                    sentence += word

                    while word != u'。':
                        x = np.zeros((1,1))
                        x[0,0] = word_num_map[word]
                        [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
                        word = to_word(probs_)
                        sentence += word


                    if len(sentence) == 2 + 2 * type:
                        sentence += u'\n'
                        poem += sentence
                        flag = False
        
    
        return poem  
    
if __name__=='__main__':
    
    tf.reset_default_graph()
    root = "/Users/liuhongbing/Documents/tensorflow/data/poetry"
    lpt = lstm_poerty_train()
    print(lpt.gen_head_poetry("刘红兵藏头", root, 5))

