import math
import os
import numpy as np
import torch
import librosa
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
    def forward_preprocess(self, batch):
        args = self.args
        data = batch
        video_path = data["video_path"]
        audio_path = data["audio_path"]
        image_path = data["first_frame_path"]
        print(
            f"[OmniTrainingModule] forward_preprocess -> video_path: {video_path}, audio_path: {audio_path}, image_path: {image_path}"
        )

        height = 360
        width = 640

        if image_path is not None:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)
            _, _, h, w = image.shape
            select_size = match_size(
                getattr(self.args, f"image_sizes_{self.args.max_hw}"), h, w
            )
            image = resize_pad(image, (h, w), select_size)
            image = image * 2.0 - 1.0
            image = image[:, :, None]
        else:
            image = None
            select_size = [height, width]
        print(
            f"selected size: {select_size}, input image size: {h}x{w}, max tokens: {args.max_tokens}"
        )

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

        # L是模型每次能生成的视频帧数，T是视频被压缩后的latent帧数
        # 这里max_tokens是写死的？对应一个经验值？16*16*4是指每个token的大小，即patch大小*通道数4
        # 所以，这里max_tokens * 16 * 16 * 4就代表了视频的总token数，除以每张图像的大小，得到视频的总帧数
        L = int(args.max_tokens * 16 * 16 * 4 / select_size[0] / select_size[1])
        print(f"video frames: {L}")

        # 让视频帧数L满足4的倍数，若L不是4的倍数，则先把 L 向下取整到最近的 4 的倍数，再加 1，保证 L 不会太小，同时让 (L + 3) // 4 依然能得到合适的 T。
        # 若L是4的倍数，则直接减3，这样 (L + 3) // 4 依然是整数，且和 latent 压缩逻辑对齐
        L = L // 4 * 4 + 1 if L % 4 != 0 else L - 3  # video frames
        print(f"video frames after adjustment: {L}")

        # T代表视频被压缩后的latent帧数，论文里说：
        # Each video, with a length T , is compressed into (T+3)/4 latent  frames using a pretrained 3D VAE, where the factor of 4 is the time compression ratio of the VAE.
        T = (L + 3) // 4  # latent frames
        print(f"latent frames: {T}")

        fixed_frame = 0
        prefix_lat_frame = 0
        first_fixed_frame = 0

        # 主要就是用Wave2Vec提音频特征，然后对齐到视频长度
        if audio_path is not None and args.use_audio:
            audio, sr = librosa.load(audio_path, sr=self.args.sample_rate)
            # 调Wav2Vec2FeatureExtractor提音频特征
            input_values = np.squeeze(
                self.wav_feature_extractor(audio, sampling_rate=16000).input_values
            )
            input_values = torch.from_numpy(input_values).float().to(device=self.device)

            # 计算音频对应的视频帧数。len(input_values) / self.args.sample_rate 得到音频的总时长（秒），再乘以 self.args.fps 得到音频对应的视频帧数
            ori_audio_len = audio_len = math.ceil(
                len(input_values) / self.args.sample_rate * self.args.fps
            )
            print(f"audio length: {audio_len} frames")
            input_values = input_values.unsqueeze(0)

            # padding audio, 扩充音频长度到满足视频帧数要求
            print(f"first_fixed_frame: {first_fixed_frame}, fixed_frame: {fixed_frame}")
            # 对齐第一段音频长度到 L - first_fixed_frame
            if audio_len < L - first_fixed_frame:
                audio_len = audio_len + (
                    (L - first_fixed_frame) - audio_len % (L - first_fixed_frame)
                )
            # 对齐后续音频段长度到 L - fixed_frame
            elif (audio_len - (L - first_fixed_frame)) % (L - fixed_frame) != 0:
                audio_len = audio_len + (
                    (L - fixed_frame)
                    - (audio_len - (L - first_fixed_frame)) % (L - fixed_frame)
                )
            input_values = F.pad(
                input_values,
                (
                    0,
                    audio_len * int(self.args.sample_rate / self.args.fps)
                    - input_values.shape[1],
                ),
                mode="constant",
                value=0,
            )

            # 用预训练的 Wav2VecModel 编码音频，得到 embedding。把所有中间层的 hidden state 拼接到一起，丰富特征表达。
            with torch.no_grad():
                hidden_states = self.audio_encoder(
                    input_values, seq_len=audio_len, output_hidden_states=True
                )
                audio_embeddings = hidden_states.last_hidden_state
                for mid_hidden_states in hidden_states.hidden_states:
                    audio_embeddings = torch.cat(
                        (audio_embeddings, mid_hidden_states), -1
                    )
            seq_len = audio_len
            print(
                f"audio embeddings shape: {audio_embeddings.shape}, seq_len: {seq_len}"
            )
            audio_embeddings = audio_embeddings.squeeze(0)
            audio_prefix = torch.zeros_like(audio_embeddings[:first_fixed_frame])
        else:
            audio_embeddings = None

        # loop
        # 分times次生成整个视频，L是模型每次能生成的视频帧数，seq_len是需要生成的视频总帧数
        # times计算逻辑：seq_len - L + first_fixed_frame是除去第一次生成的帧数后，剩余的总帧数。除以每次能生成的非overlap帧数(L - fixed_frame)，得到需要几次才能生成完，再加上第一次就是总次数
        times = (seq_len - L + first_fixed_frame) // (L - fixed_frame) + 1
        # 若没有整除，则需要多一次来生成剩余的帧数
        if times * (L - fixed_frame) + fixed_frame < seq_len:
            times += 1
        video = []
        image_emb = {}
        img_lat = None
        #
        if args.i2v:
            # 加载 VAE 并编码图片为 latent
            self.pipe.load_models_to_device(["vae"])
            img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device)
            print(
                f"img_lat shape: {img_lat.shape}, prefix_lat_frame: {prefix_lat_frame}"
            )

            msk = torch.zeros_like(img_lat.repeat(1, 1, T, 1, 1)[:, :1])
            print(f"msk shape: {msk.shape}")
            # 把图片的 latent 表示在时间维度（T，latent帧数）上复制，作为每一帧的起始 embedding。
            image_cat = img_lat.repeat(1, 1, T, 1, 1)
            print(f"image_cat shape: {image_cat.shape}")
            # 除了第一帧，后续帧的 mask都设为 1，表示这些帧需要模型去生成（第一帧用输入图片）。这个msk的作用是为了在生成时，模型知道哪些帧是需要生成的，哪些帧是输入的。
            msk[:, :, 1:] = 1
            # 把图片的 latent 表示和 mask 拼接在一起，作为视频生成模型的条件输入 embedding。
            image_emb["y"] = torch.cat([image_cat, msk], dim=1)
            print(
                f"image_emb y shape: {image_emb['y'].shape}, prefix_lat_frame: {prefix_lat_frame}"
            )

        # 开始生成视频，每一段L内并行生成，整个视频分多段串行生成。
        # 段内帧的连续性是靠模型的时序建模（如自注意力、时序卷积）和训练时的连续性损失自动保证的，推理时并行生成不会影响帧之间的自然衔接。
        for t in range(times):
            print(f"[{t+1}/{times}]")
            audio_emb = {}
            if t == 0:
                overlap = first_fixed_frame
            else:
                overlap = fixed_frame
                image_emb["y"][
                    :, -1:, :prefix_lat_frame
                ] = 0  # 第一次推理是mask只有1，往后都是mask overlap
            prefix_overlap = (3 + overlap) // 4

            # 分段生成时音频特征的分片、拼接和对齐，保证每段视频都能用到正确的音频片段，并通过 overlap 实现段与段之间的音频连续性。
            if audio_embeddings is not None:
                if t == 0:
                    audio_tensor = audio_embeddings[
                        : min(L - overlap, audio_embeddings.shape[0])
                    ]
                else:
                    audio_start = L - first_fixed_frame + (t - 1) * (L - overlap)
                    audio_tensor = audio_embeddings[
                        audio_start : min(
                            audio_start + L - overlap, audio_embeddings.shape[0]
                        )
                    ]

                audio_tensor = torch.cat([audio_prefix, audio_tensor], dim=0)
                audio_prefix = audio_tensor[-fixed_frame:]
                audio_tensor = audio_tensor.unsqueeze(0).to(
                    device=self.device, dtype=self.dtype
                )
                audio_emb["audio_emb"] = audio_tensor
            else:
                audio_prefix = None
            if image is not None and img_lat is None:
                self.pipe.load_models_to_device(["vae"])
                img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(
                    self.device
                )
                assert img_lat.shape[2] == prefix_overlap
            img_lat = torch.cat(
                [
                    img_lat,
                    torch.zeros_like(
                        img_lat[:, :, :1].repeat(1, 1, T - prefix_overlap, 1, 1)
                    ),
                ],
                dim=2,
            )
            # 调用log_video，去噪生成视频，img_lat：起始帧的 latent 表示，
            # image_emb：条件输入 embedding，有mask信息。
            # audio_emb：音频条件输入 embedding，这里的audio_emb是还没有经过AudioPack的，AudioPack是在WanModel的init里面做的，
            # prefix_overlap：前缀重叠帧数
            print(
                f"[inference]: img_lat: {img_lat.shape}, prefix_overlap: {prefix_overlap}, audio_emb: {audio_emb['audio_emb'].shape if 'audio_emb' in audio_emb else None}, image_emb y: {image_emb['y'].shape if 'y' in image_emb else None}"
            )

        inputs = {
            "latents": img_lat,
            "prompt": data["prompt"],
            "image_emb": image_emb,
            "audio_emb": audio_emb,
        }
        print(
            f"[OmniTrainingModule] forward_preprocess -> inputs keys: {inputs.keys()}"
        )

        return inputs
