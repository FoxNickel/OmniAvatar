import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from OmniAvatar.utils.args_config import parse_args

args = parse_args()


def log(str):
    if args.train_log:
        with open("train_log.txt", "a") as f:
            f.write(str + "\n")
        print(str)

def force_log(str):
    with open("train_log.txt", "a") as f:
        f.write(str + "\n")
    print(str)