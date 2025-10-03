import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from OmniAvatar.utils.args_config import parse_args

args = parse_args()

import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from OmniAvatar.datasets.datasets import WanVideoDataset
from OmniAvatar.models.training_module import OmniTrainingModule
import setproctitle
from omegaconf import OmegaConf
from OmniAvatar.utils.log import log, force_log
from pytorch_lightning.loggers import TensorBoardLogger
tb_logger = TensorBoardLogger("logs/", name="omni_avatar")
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler
import torch


def get_nested_attr(obj, attr_string):
    attrs = attr_string.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

# 自定义 collate_fn，用于过滤掉值为 None 的坏样本
def collate_fn_skip_none(batch):
    # 过滤掉所有 None 的样本
    batch = [item for item in batch if item is not None]
    # 如果过滤后 batch 为空，则返回 None 或一个空的 batch 结构
    if not batch:
        return None
    # 使用 PyTorch 默认的 collate 函数处理过滤后的 batch
    return torch.utils.data.dataloader.default_collate(batch)

def main():
    pl.seed_everything(args.seed, workers=True)
    config = OmegaConf.load(args.config)
    log(f"[train_pl.py]-main-] config: {config}")
    setproctitle.setproctitle(args.name)
    config.name = args.name
    config.savedir = os.path.join(args.savedir, args.name)
    config.batch_size = args.batch_size

    ### Define trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.savedir, "checkpoints"),
        filename="{step}",  # -{epoch:02d}
        monitor="step",
        save_last=True,
        save_top_k=-1,
        verbose=True,
        every_n_train_steps=1000,
        save_on_train_epoch_end=True,
    )

    # TODO load_full_weights的含义，以及怎么改False。省显存
    strategy = DeepSpeedStrategy(
        stage=2,
        offload_optimizer=False,
        load_full_weights=True,
        # cpu_checkpointing         =     True,
    )

    # benchmark=True。等价于 torch.backends.cudnn.benchmark=True。对于输入尺寸固定/少变的任务可加速；若每步尺寸变化大，可能反而慢或占用更多缓存。
    
    adv_profiler = AdvancedProfiler(
        dirpath=os.path.join(config.savedir, "profiler"),
        filename="advanced_profiler",
    )
    
    trainer = pl.Trainer(
        logger=tb_logger,
        default_root_dir=config.savedir,
        callbacks=[
            checkpoint_callback,
        ],  # ModelSummary(2)
        accelerator="cuda",
        # accumulate_grad_batches   =     config.gradient_accumulation_steps, 梯度累积步数。设为 k 时，会累计 k 个小 batch 的梯度再做一次优化器 step，相当于扩大有效 batch size，常用于省显存。
        benchmark=True,
        num_nodes=args.nodes,
        devices=args.devices,
        # gradient_clip_val         =     config.max_grad_norm,
        log_every_n_steps=5,
        precision=args.dtype,
        max_epochs=config.num_train_epochs,
        strategy=strategy,
        # sync_batchnorm=True, # 将 BatchNorm 转为跨 GPU 同步（SyncBatchNorm），使多卡时用全局统计量。多卡小 batch 有帮助，但会增加通信开销；若模型里几乎没有 BN 或已用 LN，可设 False。
        val_check_interval=200 if args.debug == False else 5,
        # max_steps=10, # 跑10步看profile
        profiler=adv_profiler,
        accumulate_grad_batches=4,
        # check_val_every_n_epoch   =     5,
    )

    ### Define datasets
    if args.mode == "train":
        train_dataloader = DataLoader(
            WanVideoDataset(args),
            shuffle=False,
            batch_size=config.batch_size,
            num_workers=6,
            prefetch_factor=1,
            # persistent_workers=True,
            pin_memory=False,
            timeout=60,
            drop_last=True,
            collate_fn=collate_fn_skip_none
        )
        test_dataloader = DataLoader(
            WanVideoDataset(args, validation=True),
            batch_size=config.batch_size,
            num_workers=0,
            timeout=60,
            drop_last=True,
            collate_fn=collate_fn_skip_none
        )

    # 加载要训练的模型
    trainer_model = OmniTrainingModule(args)

    # 设置可训练的模块
    # audio_encoder开train会导致音频nan，开dit看看是不是因为模块开少了导致loss为0，不对，应该是因为梯度没有导致loss=0
    trainer_model.freeze_except(["lora", "audio_proj", "audio_cond_projs", "audio_encoder"])

    log(f"===================[train_pl.py]-main-] model summary================================")
    for name, p in trainer_model.named_parameters():
        log(f"[OmniTrainingModule] - name: {name}, requires_grad: {p.requires_grad}")
    log("===================================================================================")
    
    if args.mode == "train":
        force_log(f"[OmniTrainingModule]: start training with config: {config}")
        log(f"[OmniTrainingModule]: 训练集样本数: {len(WanVideoDataset(args))}")
        log(f"[OmniTrainingModule]: 验证集样本数: {len(WanVideoDataset(args, validation=True))}")
        log(f"[OmniTrainingModule]: 训练集 batch 数: {len(train_dataloader)}")
        log(f"[OmniTrainingModule]: 验证集 batch 数: {len(test_dataloader)}")
        trainer.fit(
            model=trainer_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader,
            ckpt_path=None if not os.path.exists(args.checkpoint_path) else args.checkpoint_path,
        )

# import debugpy
# if args.local_rank == 0:
#     debugpy.listen(5678)
#     log("Waiting for debugger attach...")
#     debugpy.wait_for_client()
if __name__ == "__main__":
    main()
