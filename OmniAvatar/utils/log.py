import os
import sys
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from OmniAvatar.utils.args_config import parse_args

args = parse_args()
timestamp = datetime.datetime.now().strftime("%Y%m%d")
log_filename = f"./outputs/train_logs/train_log_{args.name}_{timestamp}.txt"

def log(str):
    if args.train_log:
        with open(log_filename, "a") as f:
            f.write(str + "\n")
        print(str)

def force_log(str):
    with open(log_filename, "a") as f:
        f.write(str + "\n")
    print(str)

def ckpt_log(str):
    if args.ckpt_log:
        with open(log_filename, "a") as f:
            f.write(str + "\n")
        print(str)