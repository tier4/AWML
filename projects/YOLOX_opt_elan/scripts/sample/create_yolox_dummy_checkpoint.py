import argparse
import os
import time

import torch
from yolox.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser("create yolox dummy checkpoint")
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    return parser


def save_checkpoint(exp_file):
    exp = get_exp(exp_file)
    model = exp.get_model()
    ckpt_state = {
        "model": model.state_dict(),
    }

    torch.save(ckpt_state, "temp_ckpt.pth")


if __name__ == "__main__":
    args = make_parser().parse_args()
    save_checkpoint(args.exp_file)
