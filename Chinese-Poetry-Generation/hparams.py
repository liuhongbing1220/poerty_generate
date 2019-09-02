import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=32000, type=int)

    # training scheme
    parser.add_argument('--_BATCH_SIZE', default=64, type=int)

    parser.add_argument('--lr', default=0.005, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=2000, type=int)

    # model
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--CHAR_VEC_DIM', default=512, type=int,
                        help="char embedding")

    parser.add_argument('--maxlen_encoder', default=27, type=int,
                        help="maximum length of a encoder sequence")

    parser.add_argument('--maxlen_decoder', default=8, type=int,
                        help="maximum length of a encoder sequence")
 
 

    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=2, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=4, type=int,
                        help="number of attention heads")
    
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

