import math
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
from scripts.inference import match_size, resize_pad


class OmniTrainingModule(pl.LightningModule):
    def __init__(self, args):
        print(f"[OmniTrainingModule] __init__")
        super().__init__()
        self.args = args

        # TODO 先去掉
        # if args.dtype == "bf16":
        #     self.dtype = torch.bfloat16
        # elif args.dtype == "fp16":
        #     self.dtype = torch.float16
        # else:
        #     self.dtype = torch.float32

        # 加载模型
        self.pipe = self.load_model()
        print(
            f"[OmniTrainingModule __init__]: Model loaded on {self.device}, dtype: {self.dtype}"
        )

        # TODO 这里要添加需要冻结的模块
        self.pipe.freeze([])

        if args.i2v:
            # 把输入的图片转换为模型需要的张量格式
            chained_trainsforms = []
            chained_trainsforms.append(TT.ToTensor())
            self.transform = TT.Compose(chained_trainsforms)

        if args.use_audio:
            # 加载音频特征提取器和音频模型
            from OmniAvatar.models.wav2vec import Wav2VecModel

            # 提特征
            self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                args.wav2vec_path
            )
            # 经过模型处理
            self.audio_encoder = Wav2VecModel.from_pretrained(
                args.wav2vec_path, local_files_only=True
            ).to(device=self.device)
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
            torch_dtype=(
                getattr(torch, args.dtype)
                if hasattr(torch, args.dtype)
                else torch.float32
            ),
            device="cpu",
        )

        pipe = WanVideoPipeline.from_model_manager(
            model_manager,
            torch_dtype=(
                getattr(torch, args.dtype)
                if hasattr(torch, args.dtype)
                else torch.float32
            ),
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

        pipe.enable_vram_management(
            num_persistent_param_in_dit=args.num_persistent_param_in_dit
        )

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

    def configure_optimizers(self):
        print(f"[OmniTrainingModule] configure_optimizers")
        return DeepSpeedCPUAdam(self.pipe.dit.parameters(), lr=float(self.args.lr))

    # Lightning里面forward主要是inference用的
    # def forward(self, data):
    #     return self.pipe.forward(data)
    # TODO 在这里调模型，算loss，训练
    def training_step(self, batch, batch_idx):
        print(
            f"[OmniTrainingModule] training_step -> batch keys: {batch.keys()}, batch_idx: {batch_idx}, batch_size: {len(batch['video_id'])}"
        )

        inputs = self.forward_preprocess(batch)
        loss = self.pipe.training_loss(**inputs)
        return loss

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
    def forward_preprocess(self, batch):
        args = self.args
        device = self.device
        max_frame = 120
        target_w, target_h = 640, 360

        video_paths = batch["video_path"]
        audio_paths = batch["audio_path"]
        prompts = batch["prompt"]
        batch_size = len(video_paths)

        videos, audios = [], []

        for i in range(batch_size):
            # 1. 读取视频
            video, _, _ = torchvision.io.read_video(video_paths[i], pts_unit="sec")
            # [T, H, W, C] -> [C, T, H, W]
            video = video.permute(3, 0, 1, 2).float() / 255.0
            # resize每帧到目标分辨率
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
                dim=1,
            )  # [C, T, H, W]

            T = video.shape[1]

            # 2. 读取音频
            audio, sr = librosa.load(audio_paths[i], sr=args.sample_rate)
            samples_per_frame = int(args.sample_rate / args.fps)

            # 3. 剪裁逻辑
            if T <= max_frame:
                video_clip = video[:, :T]
                audio_clip = audio[: T * samples_per_frame]
            else:
                start_idx = np.random.randint(0, T - max_frame + 1)
                video_clip = video[:, start_idx : start_idx + max_frame]
                audio_clip = audio[
                    start_idx * samples_per_frame : (start_idx + max_frame) * samples_per_frame
                ]

            # 4. 计算latent长度L
            T_clip = video_clip.shape[1]
            L = (T_clip + 3) // 4

            # 5. pad到4*L
            target_len = L * 4
            if T_clip < target_len:
                video_clip = F.pad(video_clip, (0, 0, 0, 0, 0, target_len - T_clip))
                audio_clip = np.pad(audio_clip, (0, (target_len - T_clip) * samples_per_frame))

            videos.append(video_clip)
            audios.append(audio_clip)

        videos = torch.stack(videos, dim=0).to(device)  # [B, C, max_frame, H, W]
        # 编码为 latent
        self.pipe.load_models_to_device(["vae"])
        with torch.no_grad():
            video_latents = self.pipe.encode_video(
                videos.to(dtype=self.dtype)
            )  # [B, latent_dim, max_frame, H', W']

        # 音频特征
        audio_embs = []
        if args.use_audio:
            for audio_clip in audios:
                input_values = np.squeeze(
                    self.wav_feature_extractor(
                        audio_clip, sampling_rate=args.sample_rate
                    ).input_values
                )
                input_values = torch.from_numpy(input_values).float().to(device)
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
        else:
            audio_embs = [None] * batch_size

        batch_inputs = {
            "input_latents": video_latents,  # [B, C, max_frame, H, W]
            "prompt": prompts,
            "audio_emb": audio_embs,
        }
        return batch_inputs
