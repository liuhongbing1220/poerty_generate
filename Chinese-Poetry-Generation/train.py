#! /usr/bin/env python3
#-*- coding:utf-8 -*-

from generate import Generator
import tensorflow as tf
from generate_transformer import Generator_transformer
from gen_transformer_v2 import Gen_transformer_v2
from plan import train_planner
from paths import save_dir
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Chinese poem generation.')
    parser.add_argument('-p', dest = 'planner', default = False, 
            action = 'store_true', help = 'train planning model')
    parser.add_argument('-g', dest = 'generator', default = False, 
            action = 'store_true', help = 'train generation model')
    parser.add_argument('-t', dest = 'generator_transformer', default = False, 
            action = 'store_true', help = 'train generation model')
    parser.add_argument('-s', dest = 'gen_transformer_v2', default = True, 
            action = 'store_true', help = 'train generation model')
    parser.add_argument('-a', dest = 'all', default = False,
            action = 'store_true', help = 'train both models')
    parser.add_argument('--clean', dest = 'clean', default = False,
            action = 'store_true', help = 'delete all models')
    
    #args = parser.parse_args()
    args=parser.parse_args(args=[])
    if args.clean:
        for f in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, f))
    else:
        if args.all or args.planner:
            print("---train-planner----")
            train_planner()
        if args.all or args.generator:
            print("---train-generator----")
            generator = Generator()
            generator.train(n_epochs = 100)
        if args.all or args.generator_transformer:
            print("---train-generator-trasformer---")
            tf.reset_default_graph()
            generator = Generator_transformer()
            generator.train(n_epochs = 1000)
        if args.all or args.gen_transformer_v2:
            print("---train-gene-trasformer-v2--")
            tf.reset_default_graph()
            generator = Gen_transformer_v2()
            generator.train(n_epochs = 1000)
        print("All training is done!")


