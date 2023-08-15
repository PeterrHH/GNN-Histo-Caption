# import os
# from simple_parsing import ArgumentParser
# import random
# from collections import defaultdict
# from pathlib import Path

# import numpy as np
# import pandas as pd

# import tensorflow as tf
# import yaml


# from dataclasses import dataclass, asdict
# from evaluation import Scorer

# import wandb

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('echo', help = 'echo the given string')
parser.add_argument('-n','--number',help = "number", type = int, default = 0, nargs = '?')
parser.add_argument('-v','--verbose',help = "Provide disc", action = "store_true") # if --verbose has value in command line, then, it is true, 
parser.add_argument('-w','--weight',help = "Wegihts", type = int, choices = [0,1,2]) # 

args = parser.parse_args()


if args.verbose:
    print(args.echo)
else:
    print("None verbose")
    
if args.weight == 0:
    print("Weight is 0")
elif args.weight == 1:
    print("Weight is 1")
else:
    print(f"{args.weight}")
    