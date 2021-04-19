"""
this will be run on the cluster when queueing jobs via log_tb1.py
simply write all arguments given here to tensorboard in the save dir, and print/log some stuff via print/logging
"""

import os
import random
from argparse import ArgumentParser
from logging import Logger, FileHandler, INFO
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    # the log_tb1 script will give us some parameters
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save_dir", type=str)
    args, unparsed = parser.parse_known_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # print statements usually go to the .out files
    print('print statement')
    print('save dir:', args.save_dir)

    # simple logging
    logger = Logger(__name__)
    logger.addHandler(FileHandler("%s/logging.txt" % args.save_dir))
    logger.setLevel(INFO)

    # log to tensorboard
    writer = SummaryWriter(log_dir="%s/tb/" % args.save_dir)

    # logging statements usually go to the .err files, and can also be saved easily
    for k, v in vars(args).items():
        print(k, v)
        logger.info("%s = %s" % (k, str(v)))
        writer.add_hparams({k: v}, {})
    for k in unparsed:
        print('unparsed:', k)
        logger.info("unparsed: %s" % k)
        writer.add_text("unparsed", k)

    # also log a random value, and add a series of random values to tensorboard
    random.seed(args.seed)
    r = random.randint(0, 1000)
    print('r:', r)
    logger.info("r: %d" % r)

    rs = [r + i for i in range(100)]
    for i, rx in enumerate(rs):
        writer.add_scalar("random/r", rx, global_step=i)
