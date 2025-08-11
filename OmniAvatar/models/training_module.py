import os
import torch
import torchvision.transforms as TT
import pytorch_lightning as pl
from transformers import Wav2Vec2FeatureExtractor
from peft import LoraConfig, inject_adapter_in_model

from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.utils.io_utils import load_state_dict
from OmniAvatar.wan_video import WanVideoPipeline


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

    # Lightning里面forward主要是inference用的
    # def forward(self, data):
    #     return self.pipe.forward(data)

    def training_step(self, batch, batch_idx):
        print(f"[OmniTrainingModule] training_step -> batch keys: {batch.keys()}")
        return None

    def configure_optimizers(self):
        print(f"[OmniTrainingModule] configure_optimizers")
        return None
