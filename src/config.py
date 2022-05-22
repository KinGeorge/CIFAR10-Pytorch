
import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='CIFAR-10 with PyTorch')
    
    parser.add_argument('--resume', action='store_true', dest='resume', default=False)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--mode', type=int, dest='mode', default=1)
    parser.add_argument('--best_acc', type=float, help='best accuracy', default=90)


    args = parser.parse_args()
    return args
