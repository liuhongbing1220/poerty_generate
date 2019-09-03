#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from char2vec import Char2Vec
from char_dict import CharDict, pad_of_sentence, start_of_sentence,end_of_sentence
from data_utils import get_batch_data
from paths import save_dir_transfomer_v2
from pron_dict import PronDict
from singleton import Singleton
from random import random
import os
import sys
import tensorflow as tf
from modules import positional_encoding, multihead_attention, ff,label_smoothing,noam_scheme,embedding
from hparams import Hparams
import numpy as np
import heapq

import logging


_model_path = os.path.join(save_dir_transfomer_v2, 'model')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

class Gen_transformer_v2(Singleton):
    
    def __init__(self,training=True):
        print('init_model_....')

        if training:
            self.context, self.sentences, self.labels, self.num_batch = get_batch_data()
        else:
            self.contexts = tf.placeholder(
                shape = [None, hp.maxlen_encoder],dtype = tf.int32, name = "context")
            self.sentences = tf.placeholder(
                shape = [None, hp.maxlen_decoder],dtype = tf.int32, name="sentences")

        self.char_dict = CharDict()
        self.char2vec = Char2Vec()
        self._build_graph()
        if not os.path.exists(save_dir_transfomer_v2):
            os.mkdir(save_dir_transfomer_v2)
        self.saver = tf.train.Saver(tf.global_variables())
        self.trained = False
        

    def _build_encoder(self, training=True):
        """ Encode context into a list of vectors. """

        self.enc = embedding(self.context, hp.vocab_size, hp.d_model,scope="dec_embed") # (N, T1, d_model)

        key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.enc), axis=-1)), -1)

        self.enc += positional_encoding(self.enc, hp.maxlen_encoder) 
        self.enc = tf.layers.dropout(self.enc, hp.dropout_rate, training=training)
        self.enc *= key_masks

        ## Blocks
        for i in range(hp.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # self-attention
                self.enc = multihead_attention(queries=self.enc,
                                          keys=self.enc,
                                          values=self.enc,
                                          num_heads=hp.num_heads,
                                          dropout_rate=hp.dropout_rate,
                                          training=training,
                                          causality=False)
                # feed forward
                self.enc = ff(self.enc, num_units=[hp.d_ff, hp.d_model])

        tf.TensorShape([hp._BATCH_SIZE, None, hp.d_model]).assert_same_rank(self.enc.shape)
        
        print('self.enc:', self.enc)

    def _build_decoder(self, training=True):
        """ Decode keyword and context into a sequence of vectors. """

        self.dec = embedding(self.sentences, hp.vocab_size, hp.d_model, scope="enc_embed") # (N, T1, d_model)
        key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.dec), axis=-1)), -1)
        self.dec += positional_encoding(self.dec, hp.maxlen_decoder)
        self.dec = tf.layers.dropout(self.dec, hp.dropout_rate, training=training)

        self.dec *= key_masks
        print('self.dec:', self.dec)

        # Blocks
        for i in range(hp.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # Masked self-attention (Note that causality is True at this time)
                self.dec = multihead_attention(queries=self.dec,
                                          keys=self.dec,
                                          values=self.dec,
                                          num_heads=hp.num_heads,
                                          dropout_rate=hp.dropout_rate,
                                          training=training,
                                          causality=True,
                                          scope="self_attention")

                # Vanilla attention
                self.dec = multihead_attention(queries=self.dec,
                                          keys=self.enc,
                                          values=self.enc,
                                          num_heads=hp.num_heads,
                                          dropout_rate=hp.dropout_rate,
                                          training=training,
                                          causality=False,
                                          scope="vanilla_attention")
                ### Feed Forward
                self.dec = ff(self.dec, num_units=[hp.d_ff, hp.d_model])
                
        print('self.dec2:', self.dec)
        tf.TensorShape([hp._BATCH_SIZE, None, hp.d_model]).assert_same_rank(self.dec.shape)



    def _build_optimizer(self):
        """ Define cross-entropy loss and minimize it. """

        self.logits = tf.layers.dense(self.dec, len(self.char_dict)) ##[None, hp.maxlen_decode, d_model]
        print('self.logits:',self.logits)
        y_smoothing = label_smoothing(tf.one_hot(self.labels, depth = len(self.char_dict)))
        print('y_smoothing:',y_smoothing)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels = y_smoothing,
                logits = self.logits)
        self.loss = tf.reduce_mean(cross_entropy)

        self.probs = tf.nn.softmax(self.logits)
        
        self.global_step = tf.Variable(tf.constant(0))
        self.learning_rate = noam_scheme(hp.lr, self.global_step, hp.warmup_steps)
#        self.learning_rate = tf.clip_by_value( tf.multiply(1.6e-5, tf.pow(2.1, self.loss)),
#                clip_value_min = 0.0002,
#                clip_value_max = 0.02)
        self.opt_step = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss = self.loss)


    def _build_graph(self):
        self._build_encoder()
        self._build_decoder()
        self._build_optimizer()

        
    def _initialize_session(self, session):
        checkpoint = tf.train.get_checkpoint_state(save_dir_transfomer_v2)
        if not checkpoint or not checkpoint.model_checkpoint_path:
            init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
            session.run(init_op)
        else:
            self.saver.restore(session, checkpoint.model_checkpoint_path)
            self.trained = True


    def generate(self, keywords):
        assert hp.NUM_OF_SENTENCES == len(keywords)
        pron_dict = PronDict()
        char_dict = CharDict()
        context = start_of_sentence()
        with tf.Session() as session:
            self._initialize_session(session)
            if not self.trained:
                print("Please train the model first! (./train.py -g)")
                sys.exit(1)
            for keyword in keywords:
                ## input context
                char = start_of_sentence()
                context_index = [char_dict.charToint(ch, 0) for ch in keyword]
                context_index2 = [char_dict.charToint(ch, 0) for ch in  context]   
                context_index.extend(context_index2)
                context_index = context+[0]*(hp.maxlen_encoder - len(context_index))

                context_index = np.reshape(context_index, (1, hp.maxlen_encoder))
                ## target context
                sentence = start_of_sentence

                for ind in range(7):
                    sentence_index = [char_dict.charToint(ch, 0) for ch in  sentence]
                    sentence_index = sentence_index + [0]*(hp.maxlen_decoder-len(sentence_index))
                    sentence_index = np.reshape(sentence_index,(1, hp.maxlen_decoder))

                    encoder_feed_dict = {
                            self.contexts : context_index,
                            self.sentences : sentence_index
                            }

                    probs = session.run(self.probs, feed_dict = encoder_feed_dict)
                    prob_list = self._gen_prob_list(probs, ind,context, pron_dict)

                    prob_sums = np.cumsum(prob_list)
                    rand_val = prob_sums[-1] * random()
                    for i, prob_sum in enumerate(prob_sums):
                        if rand_val < prob_sum:
                            char = self.char_dict.int2char(i)
                            break
                    context += char
                    sentence += char 
                context += end_of_sentence()
        return context[1:].split(end_of_sentence())


    def generate_beam_search(self, keywords, topk):
        assert hp.NUM_OF_SENTENCES == len(keywords)
        char_dict = CharDict()
        context = start_of_sentence()
        with tf.Session() as session:
            self._initialize_session(session)
            if not self.trained:
                print("Please train the model first! (./train.py -g)")
                sys.exit(1)
            for keyword in keywords:
                ## input context
                context_index = [char_dict.charToint(ch, 0) for ch in keyword]
                context_index2 = [char_dict.charToint(ch, 0) for ch in  context]   
                context_index.extend(context_index2)
                context_index = context+[0]*(hp.maxlen_encoder - len(context_index))
                context_index = np.reshape(context_index, (1, hp.maxlen_encoder))
                
                ## target context
                sentence = start_of_sentence

                ind = 0
                sentence_index = [char_dict.charToint(ch, 0) for ch in  sentence]
                sentence_index = sentence_index + [0]*(hp.maxlen_decoder-len(sentence_index))
                sentence_index = np.reshape(sentence_index,(1, hp.maxlen_decoder))
                char_topk, char_prob_topk = self._gen_char_list(session, ind,context_index, sentence_index, topk)
                sentences_topk = [sentence+ch for ch in char_topk]
                while True:

                    char_prob_list=[]
                    sentences_pred_list = []
                    ind += 1
                    for i in range(topk):
                        sentence_index = [char_dict.charToint(ch, 0) for ch in  sentences_topk[i]]
                        sentence_index = sentence_index + [0]*(hp.maxlen_decoder-len(sentence_index))
                        sentence_index = np.reshape(sentence_index,(1, hp.maxlen_decoder))

                        char_topk_tmp, char_prob_topk_tmp = self._gen_char_list(session, ind, context, context_index, sentence_index, topk)
                        char_prob_topk_tmp = char_prob_topk_tmp * char_prob_topk[i]
                        char_prob_list.extend(char_prob_topk_tmp)
                        sentences_pred_tmp = [sentences_topk[i]+ch for ch in char_topk_tmp]
                        sentences_pred_list.extend(sentences_pred_tmp)

                    char_prob_topk.clear()
                    sentences_topk.clear()
                    char_prob_topk, sentences_topk = self._get_topk_prob(char_prob_list, sentences_pred_list,topk)
                    char_prob_list.clear()
                    sentences_pred_list.clear()

                    if ind >= 7:
                        index_max = char_prob_topk.index(max(char_prob_topk))
                        context = sentences_topk[index_max]
                        context += end_of_sentence
                        break
        return context[1:].split(end_of_sentence())


    def _get_topk_prob(self,char_prob_list, sentences_pred_list,topk):

        char_topk_index = map(char_prob_list.index, heapq.nlargest(topk, char_prob_list))
        char_prob_topk = heapq.nlargest(topk,char_prob_list)
        sentences_pred_topk = [sentences_pred_list[i] for i in char_topk_index]
        return char_prob_topk, sentences_pred_topk
    
    def _gen_char_list(self, session, ind, context, context_index, sentence_index,topk):
        pron_dict = PronDict()
        encoder_feed_dict = {
                self.contexts : context_index,
                self.sentences : sentence_index,
                }

        probs = session.run(self.probs,feed_dict = encoder_feed_dict)
        prob_list = self._gen_prob_list(probs, ind, context, pron_dict)
        
        char_topk=[self.char_dict.int2char(ch) for ch in 
                                       map(prob_list.index, heapq.nlargest(topk,prob_list))]
        char_prob_topk = heapq.nlargest(topk,prob_list)

        return char_topk, char_prob_topk

    def _gen_prob_list(self, probs, ind, context, pron_dict):
        prob_list = probs.tolist()[0][ind]
        prob_list[0] = 0. ## 0 ---> pad
        prob_list[1] = 0  ## 1 ---> ^
        prob_list[-1] = 0 ## -1 ---> $
        idx = len(context)
        used_chars = set(ch for ch in context)
        for i in range(2, len(prob_list) - 1):
            ch = self.char_dict.int2char(i)
            # Penalize used characters.
            if ch in used_chars:
                prob_list[i] *= 0.4
            # Penalize rhyming violations.
            if (idx == 15 or idx == 31) and \
                    not pron_dict.co_rhyme(ch, context[7]):
                prob_list[i] *= 0.2
            # Penalize tonal violations.
            if idx > 2 and 2 == idx % 8 and \
                    not pron_dict.counter_tone(context[2], ch):
                prob_list[i] *= 0.4
            if (4 == idx % 8 or 6 == idx % 8) and \
                    not pron_dict.counter_tone(context[idx - 2], ch):
                prob_list[i] *= 0.4
        return prob_list


    def train(self, n_epochs = 6):
        print("Training transformer-based generator ...")
        with tf.Session() as session:
            self._initialize_session(session)
            ##  tf.train.shuffle_batch 必须加上队列
            coord = tf.train.Coordinator()
            _ = tf.train.start_queue_runners(session, coord)
            try:
                for epoch in range(n_epochs):
                    for batch_no in range(self.num_batch):
                        sys.stdout.write("[Seq2Seq Training] epoch = %d, line %d to %d ..." % 
                                (epoch, batch_no * hp._BATCH_SIZE,
                                (batch_no + 1) * hp._BATCH_SIZE))
                        sys.stdout.flush()
                        loss, learning_rate, _  = \
                                    session.run([self.loss, self.learning_rate, self.opt_step], 
                                    feed_dict={self.global_step: epoch*self.num_batch + batch_no})
                        print(" loss =  %f, learning_rate = %f" % (loss, learning_rate))
                        batch_no += 1
                        if 0 == batch_no % 32:
                            self.saver.save(session, _model_path)
                        self.saver.save(session, _model_path)
                        
                print("Training is done.")
            except KeyboardInterrupt:
                print("Training is interrupted.")


# For testing purpose.
if __name__ == '__main__':
    tf.reset_default_graph()
    generator = Gen_transformer_v2()
    keywords = ['四时', '变', '雪', '新']
    poem = generator.generate(keywords)
    for sentence in poem:
        print(sentence)

