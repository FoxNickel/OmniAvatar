import os
import pandas as pd
import librosa
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor
from OmniAvatar.utils.log import log, force_log

# 调用这个dataset之前，要先调用数据集下面的那个generate_metadata的脚本，生成metadata.csv
class WanVideoDataset(torch.utils.data.Dataset):
    def __init__(self, args, validation=False):
        self.args = args
        self.validation = validation
        dataset_base_path = args.dataset_base_path
        metadata_path = os.path.join(dataset_base_path, "metadata.csv")
        # metadata_path = "/home/huanglingyu/data/vgg/OmniAvatar/configs/demo.csv"
        self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.wav2vec_path
        )
        
        # 从预处理好的csv或者json里面读出来, 是个数组
        metadata = pd.read_csv(metadata_path)
        data_len = min(args.debug_data_len, len(metadata)) if args.debug else len(metadata)
        if validation:
            self.data = [metadata.iloc[i].to_dict() for i in range(data_len-10, data_len)]
            # self.data = [metadata.iloc[i].to_dict() for i in range(data_len)]
        else:
            self.data = [metadata.iloc[i].to_dict() for i in range(data_len)]
    
    def __getitem__(self, data_id):
        try:
            data = self.data[data_id % len(self.data)].copy()
            args = self.args
            
            # TODO 帧数太低，要改高，但要先解决显存过大问题
            max_frame = 25
            max_frame = max_frame // 4 * 4 + 1 if max_frame % 4 != 0 else max_frame - 3  # 对齐inference的调整
            # TODO 这里360经过vae之后，会变成360/8=45，然后进到模型之后，经过3d卷积的时候，会变成22，导致最后输出的时候跟原图h不一致。
            # 而inference的时候，h是400，是没问题的。这里要怎么处理？把原视频resize到400x640？还是说后面处理的时候补一下？
            # 先按resize到400来了
            target_w, target_h = 640, 400
            target_fps = args.fps
            
            video_path = data["video_path"]
            audio_path = data["audio_path"]
            
            # 处理视频
            video, _, info = torchvision.io.read_video(video_path, pts_unit="sec")
            origin_video_fps = info["video_fps"]
            video = video.float() / 255.0
            video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
            video = F.interpolate(video, size=(target_h, target_w), mode="bilinear", align_corners=False)
            video = video.permute(1, 0, 2, 3)  # [C, T, H, W]
            origin_video_len = video.shape[1]
            total_dur = origin_video_len / max(origin_video_fps, 1e-6)
            
            audio, sr = librosa.load(audio_path, sr=args.sample_rate)
            
            L = max_frame
            T = (L + 3) // 4
            D = L / float(target_fps)

            # 短于max的直接丢，不能扩展，扩展会让模型学错东西
            # TODO 直接在数据预处理的时候，丢掉时长小于目标帧数的视频。不能在dataset里面处理
            # 频fps不一样，导致每一帧对应多少音频采样点不一样，这里要统一。方案是：统一时间轴
            # 16000/25=640， 16000/30=533.33， 16000/24=666.67
            # 只处理音频也不行，用在线统一时间轴
            # 即：视频按原始fps读取，然后在一个统一时间网格上取帧，如果原始fps低于target_fps，就抽帧；如果高于target_fps，就复帧（最近邻），不做光流插值
            # 音频直接取对应时间窗，然后统一长度到 L * round(sr/target_fps)
            if origin_video_len <= max_frame:
                log(f"[WanVideoDataset __getitem__] -> Video shorter than max_frame, drop this video")
                return None
            
            # 起始时间：val 固定 0，train 随机
            start_sec = 0.0 if self.validation else float(np.random.uniform(0, total_dur - D))

            # 在统一时间网格上取 L 帧：抽/复帧（最近邻），不做光流插值
            t_grid = start_sec + (np.arange(L) / float(target_fps))              # [L] 秒
            frame_idxs = np.round(t_grid * origin_video_fps).astype(np.int64)    # [L]
            frame_idxs = np.clip(frame_idxs, 0, origin_video_len - 1)
            video_clip = video[:, frame_idxs, :, :]                              # [C,L,H,W]

            # 同一时间窗切音频，并固定长度到 L * round(sr/target_fps)
            a0 = int(round(start_sec * sr))
            a1 = a0 + int(round(D * sr))
            audio_clip = audio[a0:a1]

            spp_target = int(round(sr / float(target_fps)))                      # 每帧样本数（统一）
            target_audio_len = L * spp_target
            if audio_clip.shape[0] < target_audio_len:
                audio_clip = np.pad(audio_clip, (0, target_audio_len - audio_clip.shape[0]))
            else:
                audio_clip = audio_clip[:target_audio_len]

            # 音频特征提取
            audio_latent = np.squeeze(self.wav_feature_extractor(audio_clip, sampling_rate=args.sample_rate).input_values)
            audio_latent = torch.from_numpy(audio_latent)
            
            data['video'] = video_clip
            data['audio'] = audio_latent
            data['L'] = L
            data['T'] = T
            
            return data
        except Exception as e:
            # 捕获所有可能的异常，记录错误并尝试下一个样本
            force_log(f"[WanVideoDataset __getitem__] -> Error processing item {video_path}: {str(e)}")
            return None
    
    def __len__(self):
        return len(self.data)