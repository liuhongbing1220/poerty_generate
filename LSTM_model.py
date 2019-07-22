#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:08:20 2019

@author: liuhongbing
"""
import numpy as np  
import tensorflow as tf
from data_pregross import getPoetryList,DataSet,load_model

class lstm_poerty_train():
    
    def __init__(self):
        self.batch_size = 64
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
    
    #训练  
    def train_neural_network(self, xs,ys):
        
        logits, last_state, _, _, _ = self.neural_network(xs)  
        targets = tf.reshape(ys, [-1])
        
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], self.word_size)
        
        cost = tf.reduce_mean(loss)  
              
        global_step = tf.train.get_or_create_global_step()

        learing_rate = tf.train.exponential_decay(
            learning_rate=0.002, global_step=global_step, decay_steps=1000, decay_rate=0.97, staircase=True)

        optimizer = tf.train.AdamOptimizer(learing_rate)   
        
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', learing_rate)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()
        
        return loss, cost, train_op,global_step,summaries,last_state
    
    


if __name__=='__main__':
    
    tf.reset_default_graph()
    root = "/Users/liuhongbing/Documents/tensorflow/data/poetry"
    batch_size = 64
    poetry_vector,word_num_map,_— = getPoetryList(root)
    trainds = DataSet(len(poetry_vector),poetry_vector,word_num_map)

    n_chunk = len(poetry_vector) // batch_size -1


    input_data = tf.placeholder(tf.int32, [batch_size, None])  
    output_targets = tf.placeholder(tf.int32, [batch_size, None])  
    lpt = lstm_poerty_train()
    loss, cost, train_op,global_step,summaries,last_state = lpt.train_neural_network(input_data,output_targets)
    
    with tf.Session() as sess:
        
        sess.run(tf.initialize_all_variables())  
        saver = tf.train.Saver(tf.all_variables())
        last_epoch = load_model(sess, saver, root+'/model/') 
        
        print(last_epoch)
        for epoch in range(last_epoch + 1,100):
            
            print(f"iter {epoch}........................")
            all_loss = 0.0 
            for batche in range(n_chunk): 
                x,y = trainds.next_batch(batch_size)
                
                train_loss, _ , _ = sess.run([cost, last_state, train_op], feed_dict={input_data: x, output_targets: y})  

                
                all_loss = all_loss + train_loss 
                
                if batche % 50 == 1:
                    #print(epoch, batche, 0.01,train_loss) 
                    print(epoch, '\t', batche, '\t', 0.002 * (0.97 ** epoch),'\t',train_loss) 
                    
                if batche % 300 == 1:
                    saver.save(sess, root+'/model/poetry.module', global_step=epoch) 
                    print (epoch,' Loss: ', all_loss * 1.0 / n_chunk) 





