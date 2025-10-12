import os
import pandas as pd
import librosa
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor
from OmniAvatar.utils.log import log, force_log
import torch.distributed as dist
import imageio
from PIL import Image
import soundfile as sf
import cv2
import time

# 调用这个dataset之前，要先调用数据集下面的那个generate_metadata的脚本，生成metadata.csv
class WanVideoDataset(torch.utils.data.Dataset):
    def __init__(self, args, validation=False):
        self.args = args
        self.validation = validation
        dataset_base_path = args.dataset_base_path
        metadata_path = os.path.join(dataset_base_path, "metadata_10086.csv") if not args.debug else "/home/huanglingyu/data/vgg/OmniAvatar/configs/demo.csv"
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
        try:
            data = self.data[data_id % len(self.data)].copy()
            args = self.args
            rank = dist.get_rank() if dist.is_initialized() else 0
            log(f"[Dataset][rank={rank}] __getitem__ ENTER data_id={data_id} video_path={data['video_path']}")
            
            max_frame = 100
            max_frame = max_frame // 4 * 4 + 1 if max_frame % 4 != 0 else max_frame - 3  # 对齐inference的调整
            # TODO 这里360经过vae之后，会变成360/8=45，然后进到模型之后，经过3d卷积的时候，会变成22，导致最后输出的时候跟原图h不一致。
            # 而inference的时候，h是400，是没问题的。这里要怎么处理？把原视频resize到400x640？还是说后面处理的时候补一下？
            # 先按resize到400来了
            target_w, target_h = 640, 400
            target_fps = args.fps
            
            video_path = data["video_path"]
            audio_path = data["audio_path"]
            first_frame_path = data["first_frame_path"]
            
            # 处理视频
            for attempt in range(1, 4):
                try:
                    log(f"[WanVideoDataset __getitem__ info] -> try {attempt}/3 get_meta_data video_path={video_path}")
                    reader = imageio.get_reader(video_path, "ffmpeg")
                    meta = reader.get_meta_data()
                    break
                except Exception as err:
                    force_log(f"[WanVideoDataset __getitem__ warning] -> attempt {attempt}/3 load meta failed {video_path}: {err}")
                    time.sleep(0.2)
                    reader = None
            if reader is None or meta is None:
                raise RuntimeError(f"get_meta_data failed after 3 attempts for {video_path}")
            
            origin_video_fps = meta['fps']
            total_dur = meta['duration']
            origin_video_len = origin_video_fps * total_dur
            
            L = max_frame
            T = (L + 3) // 4
            D = L / float(target_fps)
            
            # 短于max的直接丢，不能扩展，扩展会让模型学错东西
            # 直接在数据预处理的时候，丢掉时长小于目标帧数的视频。不能在dataset里面处理
            if total_dur <= D:
                print(f"[WanVideoDataset __getitem__ error] -> Video shorter than max_frame, drop this video")
                reader.close()
                return None

            # 频fps不一样，导致每一帧对应多少音频采样点不一样，这里要统一。方案是：统一时间轴
            # 16000/25=640， 16000/30=533.33， 16000/24=666.67
            # 只处理音频也不行，用在线统一时间轴
            # 即：视频按原始fps读取，然后在一个统一时间网格上取帧，如果原始fps低于target_fps，就抽帧；如果高于target_fps，就复帧（最近邻），不做光流插值
            # 音频直接取对应时间窗，然后统一长度到 L * round(sr/target_fps)
            
            # 起始时间：val 固定 0，train 随机
            start_sec = 0.0 if self.validation else float(np.random.uniform(0, total_dur - D))
            # 在统一时间网格上取 L 帧：抽/复帧（最近邻），不做光流插值
            t_grid = start_sec + (np.arange(L) / float(target_fps))              # [L] 秒
            frame_idxs = np.round(t_grid * origin_video_fps).astype(np.int64)    # [L]
            frame_idxs = np.clip(frame_idxs, 0, origin_video_len - 1)
            
            # 只读取目标帧
            frames = []
            for idx in frame_idxs:
                frame = reader.get_data(int(idx))                 # [H, W, C], uint8
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                frame = torch.from_numpy(frame)                   # uint8 [H, W, C]
                frame = frame.permute(2, 0, 1).contiguous()       # [C, H, W]
                frames.append(frame)
            reader.close()
            video_clip = torch.stack(frames, dim=1)               # uint8 [C, L, H, W]

            # 同一时间窗切音频，并固定长度到 L * round(sr/target_fps)
            try:
                with sf.SoundFile(audio_path, 'r') as f:
                    src_sr = f.samplerate
                    a0 = int(round(start_sec * src_sr))
                    need = int(round(D * src_sr))
                    f.seek(max(0, a0))
                    audio_clip = f.read(frames=need, dtype='float32', always_2d=False)
                sr = src_sr
                if sr != args.sample_rate:
                    audio_clip = librosa.resample(audio_clip, orig_sr=sr, target_sr=args.sample_rate, res_type="kaiser_fast")
                    sr = args.sample_rate
            except Exception:
                audio_clip, sr = librosa.load(audio_path, sr=args.sample_rate, offset=start_sec, duration=D, mono=True)


            spp_target = int(round(sr / float(target_fps)))
            target_audio_len = L * spp_target
            if audio_clip.shape[0] < target_audio_len:
                audio_clip = np.pad(audio_clip, (0, target_audio_len - audio_clip.shape[0]))
            else:
                audio_clip = audio_clip[:target_audio_len]
            
            # 这里保存主要是为了在validation阶段，最后sample视频的时候，把裁切过后的音频和生成的视频合成一个视频
            if self.validation:
                tmp_audio_path = audio_path.replace(".wav", f"_crop.wav").replace(".mp3", f"_crop.wav")
                sf.write(tmp_audio_path, audio_clip, sr)

            # 音频特征提取
            audio_latent = np.squeeze(self.wav_feature_extractor(audio_clip, sampling_rate=args.sample_rate).input_values)
            audio_latent = torch.from_numpy(audio_latent)
            
            data['video'] = video_clip
            data['audio'] = audio_latent
            data['L'] = L
            data['T'] = T
            data['first_frame_path'] = first_frame_path
            
            log(f"[Dataset][rank={rank}] __getitem__ EXIT data_id={data_id}, video_path={data['video_path']}, data[video].shape={data['video'].shape}, data[audio].shape={data['audio'].shape}")
            return data
        except Exception as e:
            # 捕获所有可能的异常，记录错误并尝试下一个样本
            force_log(f"[WanVideoDataset __getitem__ error] -> Error processing item {video_path}: {str(e)}")
            return None
    
    def __len__(self):
        return len(self.data)