import os
import pandas as pd
import librosa
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor
from OmniAvatar.utils.log import log

# 调用这个dataset之前，要先调用数据集下面的那个generate_metadata的脚本，生成metadata.csv
class WanVideoDataset(torch.utils.data.Dataset):
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
        
        # TODO 帧数太低，要改高
        max_frame = 25
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
            log(f"[WanVideoDataset __getitem__] -> Video shorter than max_frame, drop this video")
            return None
        else:
            start_idx = np.random.randint(0, origin_video_len - max_frame + 1)
            video_clip = video[:, start_idx : start_idx + max_frame]
            audio_clip = audio[start_idx * samples_per_frame : (start_idx + max_frame) * samples_per_frame]
            log(f"[WanVideoDataset __getitem__] -> Video longer than max_frame, crop from {start_idx} to {start_idx + max_frame}")
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