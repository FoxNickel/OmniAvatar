import os
import numpy as np
import torch
import librosa
import torchvision
import torchvision.transforms as TT
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor
from peft import LoraConfig, inject_adapter_in_model

from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.utils.io_utils import load_state_dict
from OmniAvatar.wan_video import WanVideoPipeline
from deepspeed.ops.adam import DeepSpeedCPUAdam


class OmniTrainingModule(pl.LightningModule):
    def __init__(self, args):
        print(f"[OmniTrainingModule] __init__")
        super().__init__()
        self.args = args

        # 加载模型
        self.pipe = self.load_model()
        print(
            f"[OmniTrainingModule __init__]: Model loaded on {self.device}, dtype: {self.dtype}"
        )

        # 加载音频模型Wav2Vec
        from OmniAvatar.models.wav2vec import Wav2VecModel
        self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.wav2vec_path
        )
        self.audio_encoder = Wav2VecModel.from_pretrained(
            args.wav2vec_path, local_files_only=True
        ).to(device=self.device)
        self.audio_encoder.train()
        self.audio_encoder.feature_extractor._freeze_parameters()

    def load_model(self):
        # 用现有的Omni权重初始化模型
        args = self.args
        ckpt_path = f"{args.exp_path}/pytorch_model.pt"
        assert os.path.exists(
            ckpt_path
        ), f"pytorch_model.pt not found in {args.exp_path}"
        if args.train_architecture == "lora":
            pretrained_lora_path = ckpt_path
        else:
            resume_path = ckpt_path

        # 加载模型
        model_manager = ModelManager(device="cpu", infer=True)
        model_manager.load_models(
            [args.dit_path.split(","), args.text_encoder_path, args.vae_path],
            device="cpu",
        )

        pipe = WanVideoPipeline.from_model_manager(
            model_manager,
            device=self.device,  # Lightning自动分配
            use_usp=True if args.sp_size > 1 else False,
            infer=True,
        )

        if args.train_architecture == "lora":
            print(
                f"[OmniTrainingModule load_model] -> Use LoRA: lora rank: {args.lora_rank}, lora alpha: {args.lora_alpha}, pretrained_lora_path: {pretrained_lora_path}"
            )
            self.add_lora_to_model(
                pipe.denoising_model(),
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_target_modules=args.lora_target_modules,
                init_lora_weights=args.init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            missing_keys, unexpected_keys = pipe.denoising_model().load_state_dict(
                load_state_dict(resume_path), strict=True
            )
            print(
                f"[OmniTrainingModule load_model] -> load from {resume_path}, {len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys"
            )

        # TODO 待定要不要，先去掉
        # pipe.enable_vram_management(
        #     num_persistent_param_in_dit=args.num_persistent_param_in_dit
        # )

        return pipe

    def add_lora_to_model(
        self,
        model,
        lora_rank=4,
        lora_alpha=4,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        init_lora_weights="kaiming",
        pretrained_lora_path=None,
        state_dict_converter=None,
    ):
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        print(f"[OmniTrainingModule add_lora_to_model] -> lora_config: {lora_config}")
        model = inject_adapter_in_model(lora_config, model)

        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(
                f"[OmniTrainingModule add_lora_to_model] -> {num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected."
            )

    def on_fit_start(self):
        # 模型初始化的时候，device是cpu，等到fit开始的时候，Lightning会自动分配设备
        print(f"[OmniTrainingModule] on_fit_start -> device: {self.device}, param device: {next(self.parameters()).device}")
        self.pipe.device = next(self.parameters()).device
    
    def configure_optimizers(self):
        print(f"[OmniTrainingModule] configure_optimizers")
        # 这里应该是传所有需要训练的参数吧？没问题，传给Adam，但他只会更新没有freeze的
        return DeepSpeedCPUAdam(self.parameters(), lr=float(self.args.lr))

    # Lightning里面forward主要是inference用的
    # def forward(self, data):
    #     return self.pipe.forward(data)

    def training_step(self, batch, batch_idx):
        print(
            f"[OmniTrainingModule] training_step -> batch keys: {batch.keys()}, batch_idx: {batch_idx}, batch_size: {len(batch['video_id'])}"
        )

        inputs = self.forward_preprocess(batch)
        loss = self.pipe.training_loss(**inputs)
        return loss

    def freeze_except(self, trainable_model_names):
        for name, parameter in self.named_parameters():
            if any([key in name for key in trainable_model_names]):
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

    def freeze(self, frozen_model_names):
        for name, parameter in self.named_parameters():
            if any([key in name for key in frozen_model_names]):
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True

    # 这一步是对输入数据进行预处理，主要是将输入数据转换为模型需要的格式和参数。
    # 视频处理方式：最大帧数为120帧，超过的随机切片，少于的pad到120帧，分辨率640x360。
    # 音频处理方式：音频采样率为16kHz，音频长度与视频帧数对应。
    # caption先统一使用原数据集里的caption，后续可以考虑使用翻译模型进行翻译。
    # TODO check对Diffusion的理解：训练时候的输入是原视频ground truth + audio + prompt
    # TODO wave2vec看能不能用lora，看AudioPack、linear的权重大小
    # TODO cpu adam, checkpointing(deepspeed)
    # TODO 联合训练，模块打开就行
    # TODO 一定量之后validate一下（1k左右），效果在20k的时候看，batch_size=3
    # TODO CFG
    # TODO 看lora是怎么弄的
    # TODO 训练的时候，视频大小是不是要跟之前的模型保持一致
    # TODO 一定要搞清楚，模型每一步在干啥
    def forward_preprocess(self, batch):
        args = self.args
        device = self.device
        print(f"[OmniTrainingModule forward_preprocess] -> device: {device}")
        max_frame = 25
        max_frame = max_frame // 4 * 4 + 1 if max_frame % 4 != 0 else max_frame - 3  # 对齐inference的调整
        # TODO 这里360经过vae之后，会变成360/8=45，然后进到模型之后，经过3d卷积的时候，会变成22，导致最后输出的时候跟原图h不一致。
        # 而inference的时候，h是400，是没问题的。这里要怎么处理？把原视频resize到400x640？还是说后面处理的时候补一下？
        # 先按resize到400来了
        target_w, target_h = 640, 400

        video_paths = batch["video_path"]
        audio_paths = batch["audio_path"]
        prompts = batch["prompt"]
        batch_size = len(video_paths)

        videos, audios = [], []

        # 处理原视频
        for i in range(batch_size):
            print(
                f"[OmniTrainingModule forward_preprocess] -> Loading video: {video_paths[i]}"
            )
            video, _, info = torchvision.io.read_video(video_paths[i], pts_unit="sec")
            origin_video_fps = info["video_fps"]  # TODO 这里应该就是用视频本身的fps吧？
            video_fps = int(round(origin_video_fps))
            print(
                f"[OmniTrainingModule forward_preprocess] -> Original video fps: {origin_video_fps}, rounded fps: {video_fps}"
            )
            print(
                f"[OmniTrainingModule forward_preprocess] -> Raw video shape: {video.shape}"
            )  # [T, H, W, C]
            video = video.float() / 255.0
            video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
            video = torch.stack(
                [
                    F.interpolate(
                        frame.unsqueeze(0),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    for frame in video
                ],
                dim=0,  # stack回 [T, C, H, W]
            )
            video = video.permute(1, 0, 2, 3)  # [C, T, H, W]
            print(
                f"[OmniTrainingModule forward_preprocess] -> Resized video shape: {video.shape}"
            )

            origin_video_len = video.shape[1]

            print(
                f"[OmniTrainingModule forward_preprocess] -> Loading audio: {audio_paths[i]}"
            )
            audio, sr = librosa.load(audio_paths[i], sr=args.sample_rate)
            print(
                f"[OmniTrainingModule forward_preprocess] -> Raw audio shape: {audio.shape}, sr: {sr}"
            )
            samples_per_frame = int(args.sample_rate / video_fps)

            # TODO 短于max的直接丢，不能扩展，扩展会让模型学错东西
            if origin_video_len <= max_frame:
                video_clip = video[:, :origin_video_len]
                audio_clip = audio[: origin_video_len * samples_per_frame]
                print(
                    f"[OmniTrainingModule forward_preprocess] -> Video shorter than max_frame, use first {T} frames"
                )
            else:
                start_idx = np.random.randint(0, origin_video_len - max_frame + 1)
                video_clip = video[:, start_idx : start_idx + max_frame]
                audio_clip = audio[
                    start_idx
                    * samples_per_frame : (start_idx + max_frame)
                    * samples_per_frame
                ]
                print(
                    f"[OmniTrainingModule forward_preprocess] -> Video longer than max_frame, crop from {start_idx} to {start_idx + max_frame}"
                )

            L = video_clip.shape[1] # 这个L应该是=max_frame
            T = (L + 3) // 4
            print(f"[OmniTrainingModule forward_preprocess] -> L: {L}, T: {T}")
            print(
                f"[OmniTrainingModule forward_preprocess] -> Final video_clip shape: {video_clip.shape}"
            )
            print(
                f"[OmniTrainingModule forward_preprocess] -> Final audio_clip shape: {audio_clip.shape}"
            )

            # # 保存处理后的视频和音频
            # save_dir = "./debug_preprocess"
            # os.makedirs(save_dir, exist_ok=True)
            # # 保存视频为 mp4
            # video_clip_save = (video_clip.permute(1, 2, 3, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            # save_video_path = os.path.join(save_dir, f"video_clip_{i}.mp4")
            # torchvision.io.write_video(save_video_path, video_clip_save, fps=video_fps)
            # print(f"[OmniTrainingModule forward_preprocess] -> Saved video to {save_video_path}")
            # # 保存音频为wav
            # sf.write(os.path.join(save_dir, f"audio_clip_{i}.wav"), audio_clip, args.sample_rate)

            videos.append(video_clip)
            audios.append(audio_clip)

        videos = torch.stack(videos, dim=0).to(device)  # [B, C, T, H, W]
        print(
            f"[OmniTrainingModule forward_preprocess] -> All videos stacked shape: {videos.shape}, videos device: {videos.device}"
        )

        # 视频过vae
        self.pipe.load_models_to_device(["vae"])
        with torch.no_grad():
            # videos=[1, 3, 9, 360, 640]
            video_latents = self.pipe.encode_video(
                videos.to(
                    device=device, dtype=next(self.parameters()).dtype
                )  # trainer是fp16，但这里是32，要转一下，取参数类型即可
            )  # [B, latent_dim, T, H', W']
            # video_latents=[1, 16, 3, 45, 80]
        print(
            f"[OmniTrainingModule forward_preprocess] -> Latent shape: {video_latents.shape}"
        )

        # 提音频特征
        audio_embs = []
        for idx, audio_clip in enumerate(audios):
            input_values = np.squeeze(
                self.wav_feature_extractor(
                    audio_clip, sampling_rate=args.sample_rate
                ).input_values
            )
            input_values = torch.from_numpy(input_values).to(device=device, dtype=next(self.audio_encoder.parameters()).dtype)
            input_values = input_values.unsqueeze(0)
            with torch.no_grad():
                hidden_states = self.audio_encoder(
                    input_values, seq_len=max_frame, output_hidden_states=True
                )
                audio_embeddings = hidden_states.last_hidden_state
                for mid_hidden_states in hidden_states.hidden_states:
                    audio_embeddings = torch.cat(
                        (audio_embeddings, mid_hidden_states), -1
                    )
            audio_embs.append(audio_embeddings.squeeze(0))
            print(
                f"[OmniTrainingModule forward_preprocess] -> Audio embedding {idx} shape: {audio_embeddings.shape}"
            )
        # 这里将list合成batch tensor
        audio_embs = torch.stack(audio_embs, dim=0)  # [B, ...]
            
        # TODO 构造图像条件image_emb，要check逻辑
        B, C, T, H, W = video_latents.shape
        prefix_lat_frame = 1
        # 构造 image_cat: 取前prefix_lat_frame帧，repeat到T帧
        image_cat = video_latents[:, :, :prefix_lat_frame]  # [B, C, prefix_lat_frame, H, W]
        image_cat = image_cat.repeat(1, 1, T, 1, 1)    # [B, C, T, H, W]
        # 构造 mask: 已知帧为0，其余为1
        msk = torch.ones(B, 1, T, H, W, device=video_latents.device, dtype=video_latents.dtype)
        msk[:, :, :prefix_lat_frame] = 0
        # 拼接
        image_emb = {}
        image_emb['y'] = torch.cat([image_cat, msk], dim=1)  # [B, C+1, T, H, W]

        # TODO 组装数据，放到目标设备上，这里不一定要放到cuda
        target_device = device  # 或 next(self.parameters()).device
        batch_inputs = {
            "input_latents": video_latents.to(target_device),
            "image_emb": {k: v.to(target_device) for k, v in image_emb.items()},
            "prompt": prompts,
            "audio_emb": audio_embs.to(target_device) if audio_embs is not None else None,
        }
        noise = torch.randn_like(video_latents).to(target_device)
        batch_inputs["noise"] = noise
        
        print(f"[OmniTrainingModule forward_preprocess] -> batch_inputs ready, input_latents shape: {batch_inputs['input_latents'].shape}, audio_emb shape: {batch_inputs['audio_emb'].shape if batch_inputs['audio_emb'] is not None else 'None'}, noise shape: {batch_inputs['noise'].shape}")
        return batch_inputs
