import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from OmniAvatar.utils.args_config import parse_args

args = parse_args()

import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from OmniAvatar.datasets.datasets import WanVideoDataset, WanVideoValidationDataset
from OmniAvatar.models.training_module import OmniTrainingModule
import setproctitle
from omegaconf import OmegaConf


def get_nested_attr(obj, attr_string):
    attrs = attr_string.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


# TODO 看一下注释掉的东西啥意思
def main():
    pl.seed_everything(args.seed, workers=True)
    config = OmegaConf.load(args.config)
    print(f"[train_pl.py]-main-] config: {config}")
    setproctitle.setproctitle(args.name)
    config.name = args.name
    config.savedir = os.path.join(args.savedir, args.name)
    config.batch_size = args.batch_size
    # config.model.params.name = config.name
    # config.model.params.world_size = args.devices

    # obj = get_nested_attr(trainer_model, "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm3.weight")
    # obj.requires_grad = True
    ### Define trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.savedir, "checkpoints"),
        filename="{step}",  # -{epoch:02d}
        monitor="step",
        save_last=False,
        save_top_k=-1,
        verbose=True,
        every_n_train_steps=1500,
        save_on_train_epoch_end=True,
    )

    strategy = DeepSpeedStrategy(
        stage=2,
        offload_optimizer=True,
        load_full_weights=True,
        # cpu_checkpointing         =     True,
    )

    trainer = pl.Trainer(
        default_root_dir=config.savedir,
        callbacks=[
            checkpoint_callback,
        ],  # ModelSummary(2)
        accelerator="cuda",
        # accumulate_grad_batches   =     config.gradient_accumulation_steps,
        benchmark=True,
        num_nodes=args.nodes,
        devices=args.devices,
        # gradient_clip_val         =     config.max_grad_norm,
        log_every_n_steps=1,
        precision=16,
        max_epochs=config.num_train_epochs,
        strategy=strategy,
        sync_batchnorm=True,
        val_check_interval=100,
        # check_val_every_n_epoch   =     5,
    )
    # config.model.params.global_rank = trainer.global_rank
    ### Define datasets
    if args.mode == "train":
        train_dataloader = DataLoader(
            WanVideoDataset(args),
            shuffle=True,
            batch_size=config.batch_size,
            num_workers=6,
        )
        # TODO 这里拆validation数据集
        test_dataloader = DataLoader(
            WanVideoValidationDataset(args),
            batch_size=config.batch_size,
            num_workers=6,
        )

    # TODO 加载要训练的模型
    trainer_model = OmniTrainingModule(args)

    # TODO freeze module, 放到 OmniTrainingModule里
    # for name, parameter in trainer_model.named_parameters():
    #     if "video_inject" in name:
    #         parameter.requires_grad = False
    #         obj = get_nested_attr(trainer_model, name)
    #         # obj.data = obj.data / 10
    #     else:
    #         parameter.requires_grad = True

    # TODO 加载模型参数，放到 OmniTrainingModule里
    # if args.mode == "train":
    #     trainer_model.load_state_dict(
    #         torch.load("models/t2v/model.ckpt", map_location="cpu"), strict=False
    #     )
    #     # trainer_model.load_state_dict(torch.load('/home/liujianzhi/data/LVDM/logs/ST/checkpoints/step=3000.ckpt/checkpoint/mp_rank_00_model_states.pt', map_location='cpu'), strict=True)
    # else:
    #     d = torch.load(args.resume, map_location="cpu")
    #     d_con = {}
    #     for t in d["module"]:
    #         d_con[t[t.find(".") + 1 :]] = d["module"][t]
    #     trainer_model.load_state_dict(d_con, strict=False)

    # set the last linear layer of trainable model to zero
    # import torch

    # TODO 这里不要吧？
    for name, p in trainer_model.named_parameters():
        print(f"[OmniTrainingModule] - name: {name}, requires_grad: {p.requires_grad}")
        # if "inject" in name and not "P" in name and "to_out" in name:
        #     obj = get_nested_attr(trainer_model, name)
        #     obj.data.zero_()
        # if "tmp" in name:
        #     # print(name)
        #     obj = get_nested_attr(trainer_model, name)
        #     obj.data.zero_()

    if args.mode == "train":
        ### training
        print(f"[OmniTrainingModule]: start training with config: {config}")
        trainer.fit(
            model=trainer_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader,
            ckpt_path=None if not os.path.exists(args.checkpoint_path) else args.checkpoint_path,
        )


if __name__ == "__main__":
    main()
