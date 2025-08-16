import os
import torch
import time
import pytorch_lightning as pl
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
        self.audio_encoder = Wav2VecModel.from_pretrained(
            args.wav2vec_path, local_files_only=True
        ).to(device=self.device)
        # self.audio_encoder.train() # TODO就是这句导致音频nan的
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
        self.pipe.device = self.device
        # 打印模块参数统计
        self.print_module_param_report(top_n=50)
    
    def print_module_param_report(self, top_n: int = 20):
        def module_stats(mod):
            total = sum(p.numel() for p in mod.parameters())
            trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            size_mb = sum(p.numel() * p.element_size() for p in mod.parameters()) / (1024 ** 2)
            return total, trainable, size_mb

        print("[ModelStats] Top-level children of pipe:")
        for name, child in self.pipe.named_children():
            total, trainable, size_mb = module_stats(child)
            print(f"  {name:30s} params={total:,}  trainable={trainable:,}  size={size_mb:.2f} MB")

        # 列出按参数量排序的模块（包含子模块），方便定位大模块
        all_modules = []
        for name, mod in self.pipe.named_modules():
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
        print(f"[OmniTrainingModule] configure_optimizers")
        # 这里应该是传所有需要训练的参数吧？没问题，传给Adam，但他只会更新没有freeze的
        return DeepSpeedCPUAdam(self.parameters(), lr=float(self.args.lr))

    # Lightning里面forward主要是inference用的
    # def forward(self, data):
    #     return self.pipe.forward(data)

    def validation_step(self, *args, **kwargs):
        print(f"[OmniTrainingModule] validation_step, args: {args}, kwargs keys: {kwargs.keys()}")
        return super().validation_step(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        print(
            f"[OmniTrainingModule] training_step -> batch keys: {batch.keys()}, batch_idx: {batch_idx}, batch_size: {len(batch['video_id'])}"
        )

        inputs = self.forward_preprocess(batch)
        
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"[Check] {k}: nan={torch.isnan(v).any().item()}, inf={torch.isinf(v).any().item()}, shape={v.shape}")
                
        loss = self.pipe.training_loss(**inputs)
        print(f"loss: {loss.item()}, grad_fn: {loss.grad_fn}")
        
        # 检查 loss 是否为 nan 或 inf
        print(f"[Check] loss: nan={torch.isnan(loss).item()}, inf={torch.isinf(loss).item()}, value={loss.item()}")

        
        print(f"[OmniTrainingModule] training_step -> loss: {loss.item()}")
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch['video_id']))
        return loss

    def freeze_except(self, trainable_model_names):
        for name, parameter in self.named_parameters():
            if any([key in name for key in trainable_model_names]):
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False
        
        for name, module in self.named_modules():
            if any([key in name for key in trainable_model_names]):
                module.train()
            else:
                module.eval()

    def freeze(self, frozen_model_names):
        for name, parameter in self.named_parameters():
            if any([key in name for key in frozen_model_names]):
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True
        
        for name, module in self.named_modules():
            if any([key in name for key in frozen_model_names]):
                module.eval()
            else:
                module.train()

    # 这一步是对输入数据进行预处理，主要是将输入数据转换为模型需要的格式和参数。
    # 视频处理方式：最大帧数为120帧，超过的随机切片，少于的pad到120帧，分辨率640x360。
    # 音频处理方式：音频采样率为16kHz，音频长度与视频帧数对应。
    # caption先统一使用原数据集里的caption，后续可以考虑使用翻译模型进行翻译。
    # TODO check对Diffusion的理解：训练时候的输入是原视频ground truth + audio + prompt
    # TODO wave2vec看能不能用lora，看AudioPack、linear的权重大小
    # TODO 联合训练，模块打开就行
    # TODO 一定量之后validate一下（1k左右），效果在20k的时候看，batch_size=3
    # TODO CFG
    # TODO 看lora是怎么弄的
    # TODO 训练的时候，视频大小是不是要跟之前的模型保持一致
    # TODO 一定要搞清楚，模型每一步在干啥.
    # TODO 先加checkpointing
    def forward_preprocess(self, batch):
        time_start = time.time()

        prompts = batch["prompt"]
        videos = batch["video"]
        audios = batch["audio"]
        print(f"[OmniTrainingModule forward_preprocess] -> audio_path : {batch['audio_path']}, video_path: {batch['video_path']}")
        L = batch["L"][0]
        T = batch["T"][0]

        # 视频过vae
        t_vae_start = time.time()
        self.pipe.load_models_to_device(["vae"])
        with torch.no_grad():
            # videos=[1, 3, 9, 360, 640]. 这里过vae之后的video_latents在cpu上，因为vae处理逻辑把数据移到cpu了。所以to一下device
            video_latents = self.pipe.encode_video(videos).to(self.device)  # [B, latent_dim, T, H', W']
            # video_latents=[1, 16, 3, 45, 80]
        t_vae_end = time.time()
        print(f"[Timer] VAE编码耗时: {t_vae_end - t_vae_start:.3f} 秒")


        # 提音频特征
        with torch.no_grad():
            print(f"[OmniTrainingModule forward_preprocess] -> audios dtype: {audios.dtype}, device: {audios.device}, shape: {audios.shape}")
            print(f"[OmniTrainingModule forward_preprocess] -> self.audio_encoder dtype: {self.audio_encoder.dtype}, device: {self.audio_encoder.device}")
            print(f"input_values: {audios}")
            hidden_states = self.audio_encoder(audios, seq_len=L, output_hidden_states=True)
            audio_embeddings = hidden_states.last_hidden_state
            for mid_hidden_states in hidden_states.hidden_states:
                audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
            print(f"audio_embeddings {audio_embeddings}")    
        
        # TODO 构造图像条件image_emb，要check逻辑
        B, C, T, H, W = video_latents.shape
        prefix_lat_frame = 1
        image_cat = video_latents[:, :, :prefix_lat_frame]  # [B, C, prefix_lat_frame, H, W]
        image_cat = image_cat.repeat(1, 1, T, 1, 1)    # [B, C, T, H, W]
        msk = torch.ones(B, 1, T, H, W, device=video_latents.device, dtype=video_latents.dtype)
        msk[:, :, :prefix_lat_frame] = 0
        image_emb = {}
        image_emb['y'] = torch.cat([image_cat, msk], dim=1)  # [B, C+1, T, H, W]


        # 组装数据
        batch_inputs = {
            "input_latents": video_latents,
            "image_emb": {k: v for k, v in image_emb.items()},
            "prompt": prompts,
            "audio_emb": audio_embeddings,
        }
        noise = torch.randn_like(video_latents)
        batch_inputs["noise"] = noise
        
        print(f"[Timer] forward_preprocess 总耗时: {time.time() - time_start:.3f} 秒")
        print(f"[OmniTrainingModule forward_preprocess] -> batch_inputs ready, input_latents shape: {batch_inputs['input_latents'].shape}, audio_emb shape: {batch_inputs['audio_emb'].shape if batch_inputs['audio_emb'] is not None else 'None'}, noise shape: {batch_inputs['noise'].shape}")
        return batch_inputs
