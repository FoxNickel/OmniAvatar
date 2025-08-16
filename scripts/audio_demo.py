import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from OmniAvatar.utils.args_config import parse_args

args = parse_args()

import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import setproctitle
from omegaconf import OmegaConf

import os
import torch
import time
import pytorch_lightning as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam

import os
import pandas as pd
import librosa
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor


def get_nested_attr(obj, attr_string):
    attrs = attr_string.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

class AudioModule(pl.LightningModule):
    def __init__(self, args):
        print(f"[AudioModule] __init__")
        super().__init__()
        self.args = args
        
        # 加载音频模型Wav2Vec
        from OmniAvatar.models.wav2vec import Wav2VecModel
        self.audio_encoder = Wav2VecModel.from_pretrained(
            args.wav2vec_path, local_files_only=True
        ).to(device=self.device)
        # self.audio_encoder.train()
        self.audio_encoder.feature_extractor._freeze_parameters()

    def on_fit_start(self):
        # 模型初始化的时候，device是cpu，等到fit开始的时候，Lightning会自动分配设备
        print(f"[AudioModule] on_fit_start -> device: {self.device}, param device: {next(self.parameters()).device}")
        self.audio_encoder.to(device=self.device)
        # 打印模块参数统计
        self.print_module_param_report(top_n=50)
    
    def print_module_param_report(self, top_n: int = 20):
        def module_stats(mod):
            total = sum(p.numel() for p in mod.parameters())
            trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            size_mb = sum(p.numel() * p.element_size() for p in mod.parameters()) / (1024 ** 2)
            return total, trainable, size_mb

        print("[ModelStats] Top-level children of pipe:")
        for name, child in self.named_children():
            total, trainable, size_mb = module_stats(child)
            print(f"  {name:30s} params={total:,}  trainable={trainable:,}  size={size_mb:.2f} MB")

        # 列出按参数量排序的模块（包含子模块），方便定位大模块
        all_modules = []
        for name, mod in self.named_modules():
            total, trainable, size_mb = module_stats(mod)
            if total > 0:
                all_modules.append((name, total, trainable, size_mb))
        all_modules.sort(key=lambda x: x[1], reverse=True)

        print(f"[ModelStats] Top {top_n} modules by param count:")
        for name, total, trainable, size_mb in all_modules[:top_n]:
            print(f"  {name:50s} params={total:,}  trainable={trainable:,}  size={size_mb:.2f} MB")
            
        print(f"[ModelStats] Top {top_n} trainable modules by param count:")
        for name, total, trainable, size_mb in all_modules:
            if trainable > 0:
                print(f"  {name:50s} params={total:,}  trainable={trainable:,}  size={size_mb:.2f} MB")

    def configure_optimizers(self):
        print(f"[AudioModule] configure_optimizers")
        # 这里应该是传所有需要训练的参数吧？没问题，传给Adam，但他只会更新没有freeze的
        return DeepSpeedCPUAdam(self.parameters(), lr=float(self.args.lr))

    def validation_step(self, *args, **kwargs):
        print(f"[AudioModule] validation_step, args: {args}, kwargs keys: {kwargs.keys()}")
        return super().validation_step(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        print(
            f"[AudioModule] training_step -> batch keys: {batch.keys()}, batch_idx: {batch_idx}, batch_size: {len(batch['video_id'])}"
        )

        inputs = self.forward_preprocess(batch)
        
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"[Check] {k}: nan={torch.isnan(v).any().item()}, inf={torch.isinf(v).any().item()}, shape={v.shape}")
                
        # loss = self.training_loss(**inputs)
        loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # 检查 loss 是否为 nan 或 inf
        print(f"[Check] loss: nan={torch.isnan(loss).item()}, inf={torch.isinf(loss).item()}, value={loss.item()}")

        
        print(f"[AudioModule] training_step -> loss: {loss.item()}")
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch['video_id']))
        return loss

    def forward_preprocess(self, batch):
        time_start = time.time()

        audios = batch["audio"]
        audio_path = batch["audio_path"]
        video_path = batch["video_path"]
        print(f"[AudioModule forward_preprocess] -> audio_path: {audio_path}, video_path: {video_path}")
        L = batch["L"][0]
        T = batch["T"][0]

        # 提音频特征
        with torch.no_grad():
            torch.set_printoptions(profile="full")
            print(f"input_values: {audios}")
            print(f"[AudioModule forward_preprocess] -> audios dtype: {audios.dtype}, device: {audios.device}, shape: {audios.shape}")
            print(f"[AudioModule forward_preprocess] -> self.audio_encoder dtype: {self.audio_encoder.dtype}, device: {self.audio_encoder.device}, audio_len = {L}")
            hidden_states = self.audio_encoder(audios, seq_len=L, output_hidden_states=True)
            audio_embeddings = hidden_states.last_hidden_state
            for mid_hidden_states in hidden_states.hidden_states:
                audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
            print(f"audio_embeddings {audio_embeddings}")
                
            
            # audios_32 = audios.to(dtype=torch.float32)
            # audio_encoder_32 = self.audio_encoder.to(dtype=torch.float32)
            # print(f"[AudioModule forward_preprocess] -> audios_32 dtype: {audios_32.dtype}, device: {audios_32.device}, shape: {audios_32.shape}")
            # print(f"[AudioModule forward_preprocess] -> audio_encoder_32 dtype: {audio_encoder_32.dtype}, device: {audio_encoder_32.device}")
            # hidden_states_32 = audio_encoder_32(audios_32, seq_len=L, output_hidden_states=True)
            # audio_embeddings_32 = hidden_states_32.last_hidden_state
            # for mid_hidden_states_32 in hidden_states_32.hidden_states:
            #     audio_embeddings_32 = torch.cat((audio_embeddings_32, mid_hidden_states_32), -1)
            # audio_embeddings = audio_embeddings.to(dtype=self.dtype, device=self.device)
            print(f"[AudioModule forward_preprocess] -> audio_embeddings shape: {audio_embeddings.shape}, dtype: {audio_embeddings.dtype}, device: {audio_embeddings.device}")
        
        # 组装数据
        batch_inputs = {"audio_emb": audio_embeddings}
        return batch_inputs

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, args, validation=False):
        self.args = args
        self.validation = validation
        dataset_base_path = args.dataset_base_path
        metadata_path = os.path.join(dataset_base_path, "metadata.csv")
        self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.wav2vec_path
        )
        
        # 从预处理好的csv或者json里面读出来, 是个数组
        metadata = pd.read_csv(metadata_path)
        data_len = min(args.debug_data_len, len(metadata)) if args.debug else len(metadata)
        if validation:
            self.data = [metadata.iloc[i].to_dict() for i in range(data_len-10, data_len)]
        else:
            self.data = [metadata.iloc[i].to_dict() for i in range(data_len)]
    
    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        args = self.args
        
        max_frame = 106
        max_frame = max_frame // 4 * 4 + 1 if max_frame % 4 != 0 else max_frame - 3  # 对齐inference的调整
        # TODO 这里360经过vae之后，会变成360/8=45，然后进到模型之后，经过3d卷积的时候，会变成22，导致最后输出的时候跟原图h不一致。
        # 而inference的时候，h是400，是没问题的。这里要怎么处理？把原视频resize到400x640？还是说后面处理的时候补一下？
        # 先按resize到400来了
        target_w, target_h = 640, 400
        video_path = data["video_path"]
        audio_path = data["audio_path"]
        
        # 处理视频
        video, _, info = torchvision.io.read_video(video_path, pts_unit="sec")
        origin_video_fps = info["video_fps"]
        video_fps = int(round(origin_video_fps))
        video = video.float() / 255.0
        video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
        video = F.interpolate(video, size=(target_h, target_w), mode="bilinear", align_corners=False)
        video = video.permute(1, 0, 2, 3)  # [C, T, H, W]
        origin_video_len = video.shape[1]
        
        audio, sr = librosa.load(audio_path, sr=args.sample_rate)
        samples_per_frame = int(args.sample_rate / video_fps)

        # 短于max的直接丢，不能扩展，扩展会让模型学错东西
        if origin_video_len <= max_frame:
            print(f"[WanVideoDataset __getitem__] -> Video shorter than max_frame, drop this video")
            return None
        else:
            start_idx = np.random.randint(0, origin_video_len - max_frame + 1)
            video_clip = video[:, start_idx : start_idx + max_frame]
            audio_clip = audio[start_idx * samples_per_frame : (start_idx + max_frame) * samples_per_frame]
            print(f"[WanVideoDataset __getitem__] -> Video longer than max_frame, crop from {start_idx} to {start_idx + max_frame}")
        L = video_clip.shape[1] # 这个L应该是=max_frame
        T = (L + 3) // 4
        
        # 音频特征提取
        audio_latent = np.squeeze(self.wav_feature_extractor(audio_clip, sampling_rate=args.sample_rate).input_values)
        audio_latent = torch.from_numpy(audio_latent)
        
        data['video'] = video_clip
        data['audio'] = audio_latent
        data['L'] = L
        data['T'] = T
        
        return data
    
    def __len__(self):
        return len(self.data)


# TODO 看一下注释掉的东西啥意思
def main():
    pl.seed_everything(args.seed, workers=True)
    config = OmegaConf.load(args.config)
    print(f"[train_pl.py]-main-] config: {config}")
    setproctitle.setproctitle(args.name)
    config.name = args.name
    config.savedir = os.path.join(args.savedir, args.name)
    config.batch_size = args.batch_size

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
        precision=args.dtype,
        max_epochs=config.num_train_epochs,
        strategy=strategy,
        sync_batchnorm=True,
        val_check_interval=5 if args.debug else 100, # TODO 后面要改回来100
        # check_val_every_n_epoch   =     5,
    )
    # config.model.params.global_rank = trainer.global_rank
    ### Define datasets
    if args.mode == "train":
        train_dataloader = DataLoader(
            AudioDataset(args),
            shuffle=True,
            batch_size=config.batch_size,
            num_workers=6,
        )
        # TODO 这里拆validation数据集
        test_dataloader = DataLoader(
            AudioDataset(args, validation=True),
            batch_size=config.batch_size,
            num_workers=6,
        )

    # 加载要训练的模型
    trainer_model = AudioModule(args)

    # 设置可训练的模块
    # trainer_model.freeze_except(["lora", "audio_encoder", "audio_proj", "audio_cond_projs"])

    print(f"===================[train_pl.py]-main-] model summary================================")
    for name, p in trainer_model.named_parameters():
        print(f"[AudioModule] - name: {name}, requires_grad: {p.requires_grad}")
    print("===================================================================================")
    
    if args.mode == "train":
        print(f"[AudioModule]: start training with config: {config}")
        trainer.fit(
            model=trainer_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader,
            ckpt_path=None if not os.path.exists(args.checkpoint_path) else args.checkpoint_path,
        )

# import debugpy
# if args.local_rank == 0:
#     debugpy.listen(5678)
#     print("Waiting for debugger attach...")
#     debugpy.wait_for_client()
if __name__ == "__main__":
    main()