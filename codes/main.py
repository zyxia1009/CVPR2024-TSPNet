import os
from exp import Exp

if __name__ == '__main__':
    exp = Exp()
    if exp.config.mode == 'eval':
        exp.test()
    else:
        exp.train()
