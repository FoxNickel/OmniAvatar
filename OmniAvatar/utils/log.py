import os
import sys
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from OmniAvatar.utils.args_config import parse_args

args = parse_args()
timestamp = datetime.datetime.now().strftime("%Y%m%d")


def log(str):
    if args.train_log:
        with open(f"train_log_{timestamp}.txt", "a") as f:
            f.write(str + "\n")
        print(str)

def force_log(str):
    with open(f"train_log_{timestamp}.txt", "a") as f:
        f.write(str + "\n")
    print(str)

def ckpt_log(str):
    if args.ckpt_log:
        with open(f"train_log_{timestamp}.txt", "a") as f:
            f.write(str + "\n")
        print(str)