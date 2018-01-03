import torch
import argparse
import os

import model
from solver import Solver

def main(args):
    solver = Solver(root_dir = args.root_dir,
                    sub_dir = args.sub_dir,
                    batch_size = args.batch_size,
                    D_lr = args.D_lr,
                    G_lr = args.G_lr,
                    lr_decay_epoch = 5,
                    cyc_lambda = args.cyc_lambda,
                    cls_lambda = args.cls_lambda)
    
    solver.train(load_weight = args.load_weight, 
                 print_every = args.print_every)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='data',
                        help='Root dir of images')
    parser.add_argument('--sub_dir', type=str, default='train',
                        help='Subdir under the root dir. ex) subdir is train in data/train/0/male_image1.jpg')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--D_lr', type=float, default=0.0001, 
                        help='Learning rate of Discriminator')
    parser.add_argument('--G_lr', type=float, default=0.0002, 
                        help='Learning rate of Generator')
    parser.add_argument('--lr_decay_epoch', type=int, default=5,
                        help='How often you want to decay learning rate')
    parser.add_argument('--cyc_lambda', type=float, default=8, 
                        help='Weight of Cycle consistency loss')
    parser.add_argument('--cls_lambda', type=float, default=0.1, 
                        help='Weight of Gender classification loss')
    parser.add_argument('--print_every', type=int, default=400, 
                        help='How often you want to print the result')
    parser.add_argument('--load_weight', type=bool, default=False,
                        help='Load pretrained parameters')
    args = parser.parse_args()
    main(args)