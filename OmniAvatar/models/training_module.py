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
    # TODO check对Diffusion的理解：训练时候的输入是原视频ground truth + audio + prompt
    # TODO 出大问题，推理的时候长视频是通过迭代生成的，那训练的时候岂不是也只能训练L帧（L=60）的视频？
    # 那我Koala里面的视频，都有100多帧，怎么训练？拆成多个小视频训练（对，切片）？
    # 看了下还好，把大视频去掉就行。留300帧帧以内的，差不多max_tokens=7w
    # TODO L以下的就不要了，L以上的怎么切（窗口，切L出来（随机切））
    # TODO 切了之后对应的caption要处理，要不用youtube本身的title（要翻译）
    # TODO 剩下的只下audio，训audio模型
    # TODO wave2vec看能不能用lora，看AudioPack、linear的权重大小
    # TODO cpu adam, checkpointing(deepspeed)
    # TODO 联合训练，模块打开就行
    # TODO 一定量之后validate一下（1k左右），效果在20k的时候看，batch_size=3
    # TODO CFG
    def forward_preprocess(self, batch):
        args = self.args
        device = self.device
        batch_size = len(batch["video_path"])
        prompts = batch["prompt"]
        video_paths = batch["video_path"]
        audio_paths = batch["audio_path"]
        print(
            f"[OmniTrainingModule] forward_preprocess -> prompt:{prompts}, video_path: {video_paths}, audio_path: {audio_paths}"
        )

        videos = []
        for path in video_paths:
            # 读取视频为 [T, H, W, C], float32, 0~1
            video, _, _ = torchvision.io.read_video(path, pts_unit="sec")
            # 转为 [C, T, H, W]
            video = video.permute(3, 0, 1, 2)  # [C, T, H, W]
            video = video.float() / 255.0
            videos.append(video)
        # pad 到同一帧数（如训练只取前L帧，或随机裁剪L帧）
        # TODO 这里按之前说的处理，处理视频帧数
        min_frames = min([v.shape[1] for v in videos])
        # TODO 这里L应该是视频长度，要算出来的，不是这么来的。因为可能显存不够放不下。
        # 以及视频和音频要匹配才对
        L = min(args.max_frames, min_frames)  # 例如 args.max_frames=60
        videos = [v[:, :L] for v in videos]
        videos = torch.stack(videos, dim=0).to(device)  # [B, C, L, H, W]
        # 编码为 latent
        self.pipe.load_models_to_device(["vae"])
        with torch.no_grad():
            video_latents = self.pipe.encode_video(
                videos.to(dtype=self.dtype)
            )  # [B, latent_dim, L, H', W']

        # 音频特征（如需batch，可补齐/裁剪到同一长度，否则用list）
        audio_embs = []
        if args.use_audio:
            for i in range(batch_size):
                audio_path = audio_paths[i]
                audio, sr = librosa.load(audio_path, sr=args.sample_rate)
                input_values = np.squeeze(
                    self.wav_feature_extractor(audio, sampling_rate=16000).input_values
                )
                input_values = torch.from_numpy(input_values).float().to(device)
                input_values = input_values.unsqueeze(0)
                with torch.no_grad():
                    hidden_states = self.audio_encoder(
                        input_values, seq_len=L, output_hidden_states=True
                    )
                    audio_embeddings = hidden_states.last_hidden_state
                    for mid_hidden_states in hidden_states.hidden_states:
                        audio_embeddings = torch.cat(
                            (audio_embeddings, mid_hidden_states), -1
                        )
                audio_embs.append(audio_embeddings.squeeze(0))
        else:
            audio_embs = [None] * batch_size

        # TODO check视频和音频长度是否匹配
        batch_inputs = {
            "input_latents": video_latents,  # [B, C, L, H, W]
            "prompt": prompts,
            "audio_emb": audio_embs,
        }
        print(
            f"[OmniTrainingModule] forward_preprocess -> batch_inputs keys: {batch_inputs.keys()}, batch_size: {batch_size}"
        )
        print(
            f"[OmniTrainingModule] forward_preprocess -> batch_inputs input_latents shape: {batch_inputs['input_latents'].shape}, audio_emb shape: {[a.shape if a is not None else None for a in batch_inputs['audio_emb']]}"
        )
        return batch_inputs
