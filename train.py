import os
import sys
import torch
import yaml

def train(conf):
    with open(conf) as fp:
        conf = yaml.safe_load(fp)
    print(conf)

if __name__ == '__main__':
    assert len(sys.argv) == 2
    train(sys.argv[1])