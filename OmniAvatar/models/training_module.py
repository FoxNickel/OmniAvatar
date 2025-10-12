from datetime import datetime
import math
import os
import librosa
import numpy as np
import torch
import pytorch_lightning as pl
from peft import LoraConfig, inject_adapter_in_model

from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.utils.io_utils import load_state_dict
from OmniAvatar.wan_video import WanVideoPipeline
from deepspeed.ops.adam import DeepSpeedCPUAdam
from OmniAvatar.utils.log import log, force_log
import torch.distributed as dist
from OmniAvatar.utils.io_utils import save_video_as_grid_and_mp4
import torch.optim as optim
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor


class OmniTrainingModule(pl.LightningModule):
    def __init__(self, args):
        log(f"[OmniTrainingModule] __init__")
        super().__init__()
        self.args = args

        # 加载模型
        self.pipe = self.load_model()
        log(f"[OmniTrainingModule __init__]: Model loaded on {self.device}, dtype: {self.dtype}")

        # 加载音频模型Wav2Vec
        from OmniAvatar.models.wav2vec import Wav2VecModel
        self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.wav2vec_path
        )
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
            device="cpu",
        )

        pipe = WanVideoPipeline.from_model_manager(
            model_manager,
            device=self.device,  # Lightning自动分配
            use_usp=True if args.sp_size > 1 else False,
        )

        if args.train_architecture == "lora":
            log(f"[OmniTrainingModule load_model] -> Use LoRA: lora rank: {args.lora_rank}, lora alpha: {args.lora_alpha}, pretrained_lora_path: {pretrained_lora_path}")
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
            log(f"[OmniTrainingModule load_model] -> load from {resume_path}, {len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys")

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
        log(f"[OmniTrainingModule add_lora_to_model] -> lora_config: {lora_config}")
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
            log(f"[OmniTrainingModule add_lora_to_model] -> {num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")

    def on_fit_start(self):
        # 模型初始化的时候，device是cpu，等到fit开始的时候，Lightning会自动分配设备
        log(f"[OmniTrainingModule] on_fit_start -> device: {self.device}, param device: {next(self.parameters()).device}")
        self.pipe.device = self.device
        self.pipe.load_models_to_device(["vae"]) # VAE比较大，训练时也会用到，放到GPU上，只在fit开始时加载一次
        # 打印模块参数统计
        # self.print_module_param_report(top_n=50)
    
    def print_module_param_report(self, top_n: int = 20):
        def module_stats(mod):
            total = sum(p.numel() for p in mod.parameters())
            trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            size_mb = sum(p.numel() * p.element_size() for p in mod.parameters()) / (1024 ** 2)
            return total, trainable, size_mb

        log("[ModelStats] Top-level children of pipe:")
        for name, child in self.pipe.named_children():
            total, trainable, size_mb = module_stats(child)
            log(f"  {name:30s} params={total:,}  trainable={trainable:,}  size={size_mb:.2f} MB")

        # 列出按参数量排序的模块（包含子模块），方便定位大模块
        all_modules = []
        for name, mod in self.pipe.named_modules():
            total, trainable, size_mb = module_stats(mod)
            if total > 0:
                all_modules.append((name, total, trainable, size_mb))
        all_modules.sort(key=lambda x: x[1], reverse=True)

        log(f"[ModelStats] Top {top_n} modules by param count:")
        for name, total, trainable, size_mb in all_modules[:top_n]:
            log(f"  {name:50s} params={total:,}  trainable={trainable:,}  size={size_mb:.2f} MB")
            
        log(f"[ModelStats] Top {top_n} trainable modules by param count:")
        for name, total, trainable, size_mb in all_modules:
            if trainable > 0:
                log(f"  {name:50s} params={total:,}  trainable={trainable:,}  size={size_mb:.2f} MB")

    def configure_optimizers(self):
        log(f"[OmniTrainingModule] configure_optimizers")
        # 这里应该是传所有需要训练的参数吧？没问题，传给Adam，但他只会更新没有freeze的
        # return DeepSpeedCPUAdam(self.parameters(), lr=float(self.args.lr))
        return optim.AdamW(self.parameters(), lr=float(self.args.lr))

    # Lightning里面forward主要是inference用的
    # def forward(self, data):
    #     return self.pipe.forward(data)

    def validation_step(self, batch, batch_idx):
        video_path = batch['video_path']
        force_log(f"[validation_step] batch_idx={batch_idx}, video_path={video_path}, current_epoch={self.current_epoch}")
        inputs = self.forward_preprocess(batch)
        val_loss = self.pipe.training_loss(**inputs)
        self.log("val_loss", val_loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        if batch_idx == 0:
            force_log(f"[OmniTrainingModule] validation_step -> sample video_path={video_path}, current_epoch={self.current_epoch}")
            # 只在主进程做
            if dist.get_rank() == 0:
                date_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_dir = f"{self.args.savedir}/samples/samples_{video_path[0].split('/')[-2]}_{date_name}"
                sample_data = {
                    "prompt": batch["prompt"][0],
                    "image_path": batch["first_frame_path"][0],
                    "audio": batch["audio"][0],
                    "audio_path": batch["audio_path"][0],
                    "L": batch["L"][0],
                    "T": batch["T"][0],
                    "output_dir": output_dir,
                }
                sample_inputs = self.val_sample_preprocess(sample_data)
                self.sample_video(sample_inputs)

        return val_loss
    
    def val_sample_preprocess(self, data):
        prompt = data["prompt"]
        image_path = data["image_path"] # first_frame + ref image
        output_dir = data["output_dir"]
        audio = data["audio"]
        audio_path = data["audio_path"].replace(".wav", f"_crop.wav").replace(".mp3", f"_crop.wav")
        L = data["L"]
        T = data["T"]
        target_dtype = next(self.pipe.vae.parameters()).dtype
        print(f"[OmniTrainingModule] sample_preprocess -> target_dtype: {target_dtype}, device: {self.device}, prompt: {prompt}, image_path: {image_path}, output_dir: {output_dir}")
        
        # 组装audio emb
        with torch.no_grad():
            audio = audio.unsqueeze(0).to(device=self.device)
            hidden_states = self.audio_encoder(audio, seq_len=L, output_hidden_states=True)
            audio_embeddings = hidden_states.last_hidden_state
            for mid_hidden_states in hidden_states.hidden_states:
                audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
        audio_embeddings = audio_embeddings.to(dtype=target_dtype)
        audio_emb = {"audio_emb": audio_embeddings}
        
        # 组装输入图像
        target_w, target_h = 640, 400
        image = Image.open(image_path).convert("RGB")
        image = image.resize((target_w, target_h), Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div_(255.0) # [C, H, W] [3, H, W], [0, 1]
        image = image.unsqueeze(0).to(self.device) # [B, C, H, W]
        image = image * 2.0 - 1.0 # [B, C, H, W] [1, 3, H, W], [-1, 1]
        image = image[:, :, None] # [B, C, T, H, W] [1, 3, 1, H, W]
        image = image.to(dtype=target_dtype)
        
        # 组装img_lat和image_emb
        image_emb = {}
        img_lat = self.pipe.encode_video(image).to(self.device) # [B, C, 1, H', W']
        msk = torch.zeros_like(img_lat.repeat(1, 1, T, 1, 1)[:,:1])
        image_cat = img_lat.repeat(1, 1, T, 1, 1)
        msk[:, :, 1:] = 1
        image_emb["y"] = torch.cat([image_cat, msk], dim=1)
        image_emb["y"] = image_emb["y"].to(dtype=target_dtype)
        img_lat = torch.cat([img_lat, torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T - 1, 1, 1))], dim=2) # 将img_lat从1帧扩展到需要生成的段落长度，首帧是真实的，后续帧是0占位
        img_lat = img_lat.to(dtype=target_dtype)
        
        return {
            "input_latents": img_lat, # [B, C, T, H', W']
            "image_emb": image_emb,   # {y: [B, C+1, T, H', W']}
            "prompt": [prompt],       # list of str
            "audio_emb": audio_emb["audio_emb"], # [B, L, D]?
            "output_dir": output_dir,
            "audio_path": audio_path,
        }

    def sample_preprocess(self, data):
        prompt = data["prompt"]
        image_path = data["image_path"] # first_frame + ref image
        audio_path = data["audio_path"]
        output_dir = data["output_dir"]
        target_fps = self.args.fps
        target_dtype = next(self.pipe.vae.parameters()).dtype
        print(f"[OmniTrainingModule] sample_preprocess -> target_dtype: {target_dtype}, device: {self.device}, prompt: {prompt}, image_path: {image_path}, audio_path: {audio_path}, output_dir: {output_dir}")
        
        # 组装输入音频
        audio, sr = librosa.load(audio_path, sr=self.args.sample_rate)
        input_values = np.squeeze(self.wav_feature_extractor(audio, sampling_rate=16000).input_values)
        input_values = torch.from_numpy(input_values).float().to(device=self.device)
        audio_len = math.ceil(len(input_values) / self.args.sample_rate * target_fps)
        input_values = input_values.unsqueeze(0) # 这里又扩展出来batch维度了
        with torch.no_grad():
            hidden_states = self.audio_encoder(input_values, seq_len=audio_len, output_hidden_states=True)
            audio_embeddings = hidden_states.last_hidden_state
            for mid_hidden_states in hidden_states.hidden_states:
                audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
        audio_embeddings = audio_embeddings.to(dtype=target_dtype)
        audio_emb = {"audio_emb": audio_embeddings}
        
        # 组装输入图像
        target_w, target_h = 640, 400
        image = Image.open(image_path).convert("RGB")
        image = image.resize((target_w, target_h), Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div_(255.0) # [C, H, W] [3, H, W], [0, 1]
        image = image.unsqueeze(0).to(self.device) # [B, C, H, W]
        image = image * 2.0 - 1.0 # [B, C, H, W] [1, 3, H, W], [-1, 1]
        image = image[:, :, None] # [B, C, T, H, W] [1, 3, 1, H, W]
        image = image.to(dtype=target_dtype)
        
        L = audio_len
        T = (L + 3) // 4
        
        # 组装img_lat和image_emb
        image_emb = {}
        img_lat = self.pipe.encode_video(image).to(self.device) # [B, C, 1, H', W']
        msk = torch.zeros_like(img_lat.repeat(1, 1, T, 1, 1)[:,:1])
        image_cat = img_lat.repeat(1, 1, T, 1, 1)
        msk[:, :, 1:] = 1
        image_emb["y"] = torch.cat([image_cat, msk], dim=1)
        image_emb["y"] = image_emb["y"].to(dtype=target_dtype)
        img_lat = torch.cat([img_lat, torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T - 1, 1, 1))], dim=2) # 将img_lat从1帧扩展到需要生成的段落长度，首帧是真实的，后续帧是0占位
        img_lat = img_lat.to(dtype=target_dtype)
        
        return {
            "input_latents": img_lat, # [B, C, T, H', W']
            "image_emb": image_emb,   # {y: [B, C+1, T, H', W']}
            "prompt": [prompt],       # list of str
            "audio_emb": audio_emb["audio_emb"], # [B, L, D]?
            "output_dir": output_dir,
            "audio_path": audio_path,
        }
    
    @torch.no_grad()
    def sample_video(self, inputs):
        args = self.args
        img_lat = inputs["input_latents"]
        prompt = inputs["prompt"][0]
        image_emb = inputs["image_emb"]
        audio_emb = {"audio_emb": inputs["audio_emb"]}
        output_dir = inputs["output_dir"]
        audio_path = inputs["audio_path"]
        inf_steps = 25
        frames, _ = self.pipe.log_video(img_lat, prompt, 
                                           image_emb=image_emb, 
                                           audio_emb=audio_emb,
                                           num_inference_steps=inf_steps)
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)  # [1, T, C, H, W]
        save_video_as_grid_and_mp4(frames, output_dir, args.fps, audio_path=audio_path)
        
    
    def training_step(self, batch, batch_idx):
        rank = dist.get_rank() if dist.is_initialized() else 0
        log(f"[training_step][rank={rank}] batch_idx={batch_idx} video_path={batch['video_path']}")
        # 打印 batch 主要字段 shape
        # for k, v in batch.items():
        #     if isinstance(v, torch.Tensor):
        #         log(f"[Rank {rank}] batch[{k}] shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")

        inputs = self.forward_preprocess(batch)
        
        # for k, v in inputs.items():
        #     if isinstance(v, torch.Tensor):
        #         log(f"[Check] {k}: nan={torch.isnan(v).any().item()}, inf={torch.isinf(v).any().item()}, shape={v.shape}")
                
        loss = self.pipe.training_loss(**inputs)
        # log(f"loss: {loss.item()}, grad_fn: {loss.grad_fn}")
        
        # 检查 loss 是否为 nan 或 inf
        # log(f"[Check] loss: nan={torch.isnan(loss).item()}, inf={torch.isinf(loss).item()}, value={loss.item()}")

        
        log(f"[OmniTrainingModule] training_step -> loss: {loss.item()}, video_path={batch['video_path']}")
        # 合并成一行，同时支持进度条和所有 loggers（包括 TensorBoard）
        self.log("train_loss", loss,
                prog_bar=True,           # 进度条显示
                on_step=True,            # 每步记录
                on_epoch=True,           # 每轮记录
                batch_size=len(batch['video_id']),  # batch size
                logger=True,             # 发送到所有 loggers（TensorBoard/W&B等）
                sync_dist=True)          # 多卡时同步
        
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
    # TODO wave2vec看能不能用lora，看AudioPack、linear的权重大小
    # TODO 一定量之后validate一下（1k左右），效果在20k的时候看，batch_size=1
    # TODO 训练的时候，视频大小是不是要跟之前的模型保持一致
    # TODO 一定要搞清楚，模型每一步在干啥.
    def forward_preprocess(self, batch):
        # time_start = time.time()

        prompts = batch["prompt"]
        videos = batch["video"]
        audios = batch["audio"]
        log(f"[OmniTrainingModule forward_preprocess] start-> audio_path : {batch['audio_path']}, video_path: {batch['video_path']}")
        L = batch["L"][0]
        T = batch["T"][0]

        # 视频过vae
        # t_vae_start = time.time()
        # self.pipe.load_models_to_device(["vae"])
        # videos=[1, 3, 9, 360, 640]. 这里过vae之后的video_latents在cpu上，因为vae处理逻辑把数据移到cpu了。所以to一下device
        videos = videos.to(self.device, non_blocking=True).to(torch.float16).div_(255.0)
        video_latents = self.pipe.encode_video(videos).to(self.device)  # [B, latent_dim, T, H', W']
        # video_latents=[1, 16, 3, 45, 80]
        # t_vae_end = time.time()
        # log(f"[Timer] VAE编码耗时: {t_vae_end - t_vae_start:.3f} 秒")


        # 提音频特征
        # log(f"[OmniTrainingModule forward_preprocess] -> audios dtype: {audios.dtype}, device: {audios.device}, shape: {audios.shape}")
        # log(f"[OmniTrainingModule forward_preprocess] -> self.audio_encoder dtype: {self.audio_encoder.dtype}, device: {self.audio_encoder.device}")
        # log(f"input_values: {audios}")
        hidden_states = self.audio_encoder(audios, seq_len=L, output_hidden_states=True)
        audio_embeddings = hidden_states.last_hidden_state
        for mid_hidden_states in hidden_states.hidden_states:
            audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
        # log(f"audio_embeddings {audio_embeddings}")    
        
        # TODO 构造图像条件image_emb，要check逻辑
        B, C, T, H, W = video_latents.shape
        prefix_lat_frame = 1
        image_cat = video_latents[:, :, :prefix_lat_frame]  # [B, C, prefix_lat_frame, H, W]
        # TODO 这里好像漏了image = image * 2.0 - 1.0归一化到[-1, 1]？？？
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
        
        # 确保输入有梯度
        batch_inputs["input_latents"] = batch_inputs["input_latents"].detach().requires_grad_(True)
        batch_inputs["noise"] = batch_inputs["noise"].detach().requires_grad_(True)
        
        # log(f"[Timer] forward_preprocess 总耗时: {time.time() - time_start:.3f} 秒")
        log(f"[OmniTrainingModule forward_preprocess end] ->video_path={batch['video_path']}, batch_inputs ready, input_latents shape: {batch_inputs['input_latents'].shape}, audio_emb shape: {batch_inputs['audio_emb'].shape if batch_inputs['audio_emb'] is not None else 'None'}, noise shape: {batch_inputs['noise'].shape}")
        return batch_inputs
