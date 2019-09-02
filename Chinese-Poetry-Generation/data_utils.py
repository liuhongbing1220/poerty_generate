#! /usr/bin/env python3
#-*- coding:utf-8 -*-

from char_dict import end_of_sentence, start_of_sentence,CharDict
from paths import gen_data_path, plan_data_path, check_uptodate
from poems import Poems
from rank_words import RankedWords
from segment import Segmenter
from hparams import Hparams
import tensorflow as tf
import re
import subprocess

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

def gen_train_data():
    print("Generating training data ...")
    segmenter = Segmenter()
    poems = Poems()
    poems.shuffle()
    ranked_words = RankedWords()
    plan_data = []
    gen_data = []
    for poem in poems:
        if len(poem) != 4:
            continue # Only consider quatrains.
        valid = True
        context = start_of_sentence()
        gen_lines = []
        keywords = []
        for sentence in poem:
            if len(sentence) != 7:
                valid = False
                break
            words = list(filter(lambda seg: seg in ranked_words, segmenter.segment(sentence)))
            
            if len(words) == 0:
                valid = False
                break
            
            keyword = words[0]
            for word in words[1 : ]:
                if ranked_words.get_rank(word) < ranked_words.get_rank(keyword):
                    keyword = word
                    
            gen_line = sentence + end_of_sentence() + \
                    '\t' + keyword + '\t' + context + '\n'
            gen_lines.append(gen_line)
            keywords.append(keyword)
            context += sentence + end_of_sentence()
        if valid:
            plan_data.append('\t'.join(keywords) + '\n')
            gen_data.extend(gen_lines)
            
    with open(plan_data_path, 'w') as fout:
        for line in plan_data:
            fout.write(line)
            
    with open(gen_data_path, 'w') as fout:
        for line in gen_data:
            fout.write(line)


def batch_train_data(batch_size):
    """ Training data generator for the poem generator."""
    gen_train_data() # Shuffle data order and cool down CPU.
    keywords = []
    contexts = []
    sentences = []
    with open(gen_data_path, 'r') as fin:
        for line in fin.readlines():
            toks = line.strip().split('\t')
            sentences.append(toks[0])
            keywords.append(toks[1])
            contexts.append(toks[2])
            if len(keywords) == batch_size:
                yield keywords, contexts, sentences
                keywords.clear()
                contexts.clear()
                sentences.clear()
        # For simplicity, only return full batches for now.


def load_train_data_embedding():
    import numpy as np
    """ Training data generator for the poem generator."""
    gen_train_data() # Shuffle data order and cool down CPU.
    char_dict = CharDict()
    contexts = []
    sentences = []
    targets = []
    with open(gen_data_path, 'r') as fin:
        for line in fin.readlines():
            toks = line.strip().split('\t')
            sent1 = [1] ## 添加开始字符
            sent2 = [char_dict.charToint(ch, 0) for ch in toks[0]]
            cont1 = [char_dict.charToint(ch, 0) for ch in toks[1]]
            cont2 = [char_dict.charToint(ch, 0) for ch in toks[2]]
            sent1.extend(sent2)
            cont1.extend(cont2)
            if len(cont1) > hp.maxlen_encoder or len(sent1) > hp.maxlen_decoder+1:
                continue
            contexts.append(np.array(cont1))
            sentences.append(np.array(sent1[:-1]))
            targets.append(np.array(sent1[1:]))

    X = np.zeros([len(contexts), hp.maxlen_encoder], np.int32)
    Y = np.zeros([len(sentences), hp.maxlen_decoder], np.int32)
    labels = np.zeros([len(sentences), hp.maxlen_decoder], np.int32)
    for i, (x, y, z) in enumerate(zip(contexts, sentences, targets)):
#        print('hp.maxlen_encoder-len(contexts[i]):',hp.maxlen_encoder-len(contexts[i]))
#        print('hp.maxlen_decoder-len(sentences[i])]:',hp.maxlen_decoder-len(sentences[i]))
        X[i] = np.lib.pad(x, [0, hp.maxlen_encoder-len(contexts[i])], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen_decoder-len(sentences[i])], 'constant', constant_values=(0, 0))
        labels[i] = np.lib.pad(z, [0, hp.maxlen_decoder-len(targets[i])], 'constant', constant_values=(0, 0))

    return X, Y, labels


def get_batch_data():
    # Load data
    X, Y, labels = load_train_data_embedding()    
    print("Load %d pairs of couplet." % (len(X)))
    # calc total batch count
    num_batch = len(X) // hp._BATCH_SIZE
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    labels= tf.convert_to_tensor(labels, tf.int32)

    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y, labels])

    # create batch queues
    x, y, labels = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp._BATCH_SIZE, 
                                capacity=hp._BATCH_SIZE*64,   
                                min_after_dequeue=hp._BATCH_SIZE*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, labels, num_batch # (N, T), (N, T), ()




def batch_train_data_transformer(batch_size):
    """ Training data generator for the poem generator."""
    gen_train_data() # Shuffle data order and cool down CPU.
    contexts = []
    sentences = []
    with open(gen_data_path, 'r') as fin:
        for line in fin.readlines():
            toks = line.strip().split('\t')
            sentences.append(toks[0])
            cont = toks[1]+toks[2]
            if len(cont)>27:
                cont = cont[:27]
            contexts.append(cont)
            if len(contexts) == batch_size:
                yield contexts, sentences
                contexts.clear()
                sentences.clear()





if __name__ == '__main__':
    if not check_uptodate(plan_data_path) or \
            not check_uptodate(gen_data_path):
        gen_train_data()

