#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from char2vec import Char2Vec
from char_dict import CharDict, pad_of_sentence, start_of_sentence
from data_utils import batch_train_data_transformer
from paths import save_dir
from singleton import Singleton
import numpy as np
import os
import sys
import tensorflow as tf
from modules import positional_encoding, multihead_attention, ff,label_smoothing,noam_scheme
from hparams import Hparams
import logging


_model_path = os.path.join(save_dir, 'model')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

class Generator_transformer(Singleton):


    def _build_encoder(self, training=True):
        """ Encode context into a list of vectors. """
        self.context = tf.placeholder(
                shape = [hp._BATCH_SIZE, None, hp.CHAR_VEC_DIM],
                dtype = tf.float32, 
                name = "context")
        
        self.enc = self.context # (N, T1, d_model)
        self.enc *= hp.d_model**0.5 # scale
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
        self.decoder_inputs = tf.placeholder(
                shape = [hp._BATCH_SIZE, None, hp.CHAR_VEC_DIM],
                dtype = tf.float32, 
                name = "decoder_inputs")


        self.dec = self.decoder_inputs
        self.dec *= hp.d_model ** 0.5  # scale
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
        self.targets = tf.placeholder(
                shape = [None],
                dtype = tf.int32, 
                name = "targets")

        self.logits = tf.layers.dense(self.dec, len(self.char_dict))
        print('self.logits:',self.logits)
        y_smoothing = label_smoothing(tf.one_hot(self.targets, depth = len(self.char_dict)))
        print('y_smoothing:',y_smoothing)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels = y_smoothing,
                logits = self.logits)
        self.loss = tf.reduce_mean(cross_entropy)

        self.probs = tf.nn.softmax(self.logits)
        
        global_step = tf.train.get_or_create_global_step()
        self.learning_rate = noam_scheme(hp.lr, global_step, hp.warmup_steps)
#        self.learning_rate = tf.clip_by_value( tf.multiply(1.6e-5, tf.pow(2.1, self.loss)),
#                clip_value_min = 0.0002,
#                clip_value_max = 0.02)
        
        self.opt_step = tf.train.AdamOptimizer(
                learning_rate = self.learning_rate).\
                        minimize(loss = self.loss)

    def _build_graph(self):
        self._build_encoder()
        self._build_decoder()
        self._build_optimizer()

    def __init__(self):
        self.char_dict = CharDict()
        self.char2vec = Char2Vec()
        self._build_graph()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.saver = tf.train.Saver(tf.global_variables())
        self.trained = False
        
    def _initialize_session(self, session):
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        if not checkpoint or not checkpoint.model_checkpoint_path:
            init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
            session.run(init_op)
        else:
            self.saver.restore(session, checkpoint.model_checkpoint_path)
            self.trained = True


    def _gen_prob_list(self, probs, context, pron_dict):
        prob_list = probs.tolist()[0]
        prob_list[0] = 0
        prob_list[-1] = 0
        idx = len(context)
        used_chars = set(ch for ch in context)
        for i in range(1, len(prob_list) - 1):
            ch = self.char_dict.int2char(i)
            # Penalize used characters.
            if ch in used_chars:
                prob_list[i] *= 0.6
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
        print("Training RNN-based generator ...")
        with tf.Session() as session:
            self._initialize_session(session)
            try:
                for epoch in range(n_epochs):
                    batch_no = 0
                    for contexts, sentences in batch_train_data_transformer(hp._BATCH_SIZE):
                        sys.stdout.write("[Seq2Seq Training] epoch = %d, " \
                                "line %d to %d ..." % 
                                (epoch, batch_no * hp._BATCH_SIZE,
                                (batch_no + 1) * hp._BATCH_SIZE))
                        sys.stdout.flush()

                        self._train_a_batch(session, epoch, contexts, sentences)
                        batch_no += 1
                        if 0 == batch_no % 32:
                            self.saver.save(session, _model_path)
                    self.saver.save(session, _model_path)
                print("Training is done.")
            except KeyboardInterrupt:
                print("Training is interrupted.")

    def _train_a_batch(self, session, epoch, contexts, sentences):
        context_data = self._embedding(contexts, hp.maxlen_encoder)
        decoder_inputs = self._embedding([start_of_sentence() + sentence[:-1] for sentence in sentences], hp.maxlen_decoder)
        
        targets = self._fill_targets(sentences)
        feed_dict = {
                self.context : context_data,
                self.decoder_inputs : decoder_inputs,
                self.targets : targets
                }
        loss, learning_rate, _ = session.run(
                [self.loss, self.learning_rate, self.opt_step],
                feed_dict = feed_dict)
        print(" loss =  %f, learning_rate = %f" % (loss, learning_rate))


    def _embedding(self, texts, maxtime):
        
        matrix = np.zeros([hp._BATCH_SIZE, maxtime, hp.CHAR_VEC_DIM], dtype = np.float32)
        for i in range(hp._BATCH_SIZE):
            for j in range(maxtime):
                matrix[i, j, :] = self.char2vec.get_vect(pad_of_sentence())
                
        for i, text in enumerate(texts):
            matrix[i, : len(text)] = self.char2vec.get_vects(text)
        return matrix

    def _fill_targets(self, sentences):
        targets = []
        for sentence in sentences:
            targets.extend(map(self.char_dict.char2int, sentence))
        return targets


# For testing purpose.
if __name__ == '__main__':
    tf.reset_default_graph()
    generator = Generator()
    keywords = ['四时', '变', '雪', '新']
    poem = generator.generate(keywords)
    for sentence in poem:
        print(sentence)

