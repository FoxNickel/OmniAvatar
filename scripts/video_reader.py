import librosa
import torch
import numpy as np
import imageio
from PIL import Image
import time
import torchvision
import torch.nn.functional as F
from moviepy import ImageSequenceClip, AudioFileClip
import soundfile as sf

# def torchvision_save():
#     try:
#         data = self.data[data_id % len(self.data)].copy()
#         args = self.args
#         rank = dist.get_rank() if dist.is_initialized() else 0
#         log(f"[Dataset][rank={rank}] __getitem__ ENTER data_id={data_id} video_path={data['video_path']}")
        
#         # TODO 帧数太低，要改高，但要先解决显存过大问题
#         max_frame = 75
#         max_frame = max_frame // 4 * 4 + 1 if max_frame % 4 != 0 else max_frame - 3  # 对齐inference的调整
#         # TODO 这里360经过vae之后，会变成360/8=45，然后进到模型之后，经过3d卷积的时候，会变成22，导致最后输出的时候跟原图h不一致。
#         # 而inference的时候，h是400，是没问题的。这里要怎么处理？把原视频resize到400x640？还是说后面处理的时候补一下？
#         # 先按resize到400来了
#         target_w, target_h = 640, 400
#         target_fps = args.fps
        
#         video_path = data["video_path"]
#         audio_path = data["audio_path"]
        
#         # 处理视频
#         video, _, info = torchvision.io.read_video(video_path, pts_unit="sec")
#         origin_video_fps = info["video_fps"]
#         video = video.float() / 255.0
#         video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
#         video = F.interpolate(video, size=(target_h, target_w), mode="bilinear", align_corners=False)
#         video = video.permute(1, 0, 2, 3)  # [C, T, H, W]
#         origin_video_len = video.shape[1]
#         total_dur = origin_video_len / max(origin_video_fps, 1e-6)
        
#         L = max_frame
#         T = (L + 3) // 4
#         D = L / float(target_fps)

#         # 频fps不一样，导致每一帧对应多少音频采样点不一样，这里要统一。方案是：统一时间轴
#         # 16000/25=640， 16000/30=533.33， 16000/24=666.67
#         # 只处理音频也不行，用在线统一时间轴
#         # 即：视频按原始fps读取，然后在一个统一时间网格上取帧，如果原始fps低于target_fps，就抽帧；如果高于target_fps，就复帧（最近邻），不做光流插值
#         # 音频直接取对应时间窗，然后统一长度到 L * round(sr/target_fps)
        
#         # 起始时间：val 固定 0，train 随机
#         start_sec = 0.0 if self.validation else float(np.random.uniform(0, total_dur - D))
#         # 在统一时间网格上取 L 帧：抽/复帧（最近邻），不做光流插值
#         t_grid = start_sec + (np.arange(L) / float(target_fps))              # [L] 秒
#         frame_idxs = np.round(t_grid * origin_video_fps).astype(np.int64)    # [L]
#         frame_idxs = np.clip(frame_idxs, 0, origin_video_len - 1)
#         video_clip = video[:, frame_idxs, :, :]                              # [C,L,H,W]

#         # 同一时间窗切音频，并固定长度到 L * round(sr/target_fps)
#         audio, sr = librosa.load(audio_path, sr=args.sample_rate)
#         a0 = int(round(start_sec * sr))
#         a1 = a0 + int(round(D * sr))
#         audio_clip = audio[a0:a1]

#         spp_target = int(round(sr / float(target_fps)))                      # 每帧样本数（统一）
#         target_audio_len = L * spp_target
#         if audio_clip.shape[0] < target_audio_len:
#             audio_clip = np.pad(audio_clip, (0, target_audio_len - audio_clip.shape[0]))
#         else:
#             audio_clip = audio_clip[:target_audio_len]

#         # 音频特征提取
#         audio_latent = np.squeeze(self.wav_feature_extractor(audio_clip, sampling_rate=args.sample_rate).input_values)
#         audio_latent = torch.from_numpy(audio_latent)
        
#         data['video'] = video_clip
#         data['audio'] = audio_latent
#         data['L'] = L
#         data['T'] = T
        
#         log(f"[Dataset][rank={rank}] __getitem__ EXIT data_id={data_id}, video_path={data['video_path']}, data[video].shape={data['video'].shape}, data[audio].shape={data['audio'].shape}")
#         return data
#     except Exception as e:
#         # 捕获所有可能的异常，记录错误并尝试下一个样本
#         force_log(f"[WanVideoDataset __getitem__ error] -> Error processing item {video_path}: {str(e)}")
#         return None

def read_video_by_imageio(video_path, audio_path):
    save_path="output_imageio.mp4"
    
    start_time = time.time()
    max_frame = 100
    max_frame = max_frame // 4 * 4 + 1 if max_frame % 4 != 0 else max_frame - 3  # 对齐inference的调整
    target_w, target_h = 640, 400
    target_fps = 25
    
    
    reader = imageio.get_reader(video_path, 'ffmpeg')
    meta = reader.get_meta_data()
    origin_video_fps = meta['fps']
    total_dur = meta['duration']
    origin_video_len = origin_video_fps * total_dur

    L = max_frame
    T = (L + 3) // 4
    D = L / float(target_fps)
    
    if total_dur <= D:
        print(f"[WanVideoDataset __getitem__ error] -> Video shorter than max_frame, drop this video")
        reader.close()
        return None

    start_sec = float(np.random.uniform(0, total_dur - D))
    t_grid = start_sec + (np.arange(L) / float(target_fps))              # [L] 秒
    frame_idxs = np.round(t_grid * origin_video_fps).astype(np.int64)    # [L]
    frame_idxs = np.clip(frame_idxs, 0, origin_video_len - 1)
    
    # 只读取目标帧
    frames = []
    for idx in frame_idxs:
        idx = int(idx)
        frame = reader.get_data(idx)  # ndarray [H, W, C], uint8
        frame = Image.fromarray(frame)
        frame = frame.resize((target_w, target_h), Image.BILINEAR)
        frame = np.asarray(frame).astype(np.float32) / 255.0  # [H, W, C], float32
        frame = torch.from_numpy(frame).permute(2, 0, 1)      # [C, H, W]
        frames.append(frame)
    reader.close()
    video_clip = torch.stack(frames, dim=1)  # [C, L, H, W]
    
    # 保存视频帧为 mp4
    frames_np = [ (frame.permute(1,2,0).numpy() * 255).astype(np.uint8) for frame in frames ]  # [H,W,C], uint8
    clip = ImageSequenceClip(frames_np, fps=target_fps)

    # 同一时间窗切音频，并固定长度到 L * round(sr/target_fps)
    audio, sr = librosa.load(audio_path, sr=16000)
    a0 = int(round(start_sec * sr))
    a1 = a0 + int(round(D * sr))
    audio_clip = audio[a0:a1]

    spp_target = int(round(sr / float(target_fps)))                      # 每帧样本数（统一）
    target_audio_len = L * spp_target
    if audio_clip.shape[0] < target_audio_len:
        audio_clip = np.pad(audio_clip, (0, target_audio_len - audio_clip.shape[0]))
    else:
        audio_clip = audio_clip[:target_audio_len]
    
    # 保存临时音频
    tmp_audio_path = "tmp_audio.wav"
    sf.write(tmp_audio_path, audio_clip, sr)

    # 合成音视频
    audio_clip_moviepy = AudioFileClip(tmp_audio_path)
    clip = clip.with_audio(audio_clip_moviepy)
    clip.write_videofile(save_path, codec="libx264", audio_codec="aac")
    print(f"保存合成视频到: {save_path}")
    
    print(f"imageio数据处理耗时：{time.time() - start_time:.3f} 秒")
    print(f"[Dataset] __getitem__ EXIT data[video].shape={video_clip.shape}, data[audio].shape={audio_clip.shape}")
        

def read_video_by_torchvision(video_path, audio_path):
    start_time = time.time()
    max_frame = 100
    max_frame = max_frame // 4 * 4 + 1 if max_frame % 4 != 0 else max_frame - 3  # 对齐inference的调整
    target_w, target_h = 640, 400
    target_fps = 25
    
    # 处理视频
    video, _, info = torchvision.io.read_video(video_path, pts_unit="sec")
    origin_video_fps = info["video_fps"]
    video = video.float() / 255.0
    video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
    video = F.interpolate(video, size=(target_h, target_w), mode="bilinear", align_corners=False)
    video = video.permute(1, 0, 2, 3)  # [C, T, H, W]
    origin_video_len = video.shape[1]
    total_dur = origin_video_len / max(origin_video_fps, 1e-6)
    
    audio, sr = librosa.load(audio_path, sr=16000)
    
    L = max_frame
    T = (L + 3) // 4
    D = L / float(target_fps)

    if origin_video_len <= max_frame:
        print(f"[WanVideoDataset __getitem__] -> Video shorter than max_frame, drop this video")
        return None
    
    # 起始时间：val 固定 0，train 随机
    start_sec = float(np.random.uniform(0, total_dur - D))

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
    
    print(f"torchvision数据处理耗时：{time.time() - start_time:.3f} 秒")
    print(f"[Dataset] __getitem__ EXIT data[video].shape={video_clip.shape}, data[audio].shape={audio_clip.shape}")

if __name__ == "__main__":
    video_path = "/mnt/hdd2/huanglingyu/vgg/datasets/Koala-36M-v1/videos/Koala_36M_3_sv/pv97fCP13SE_68/video.mp4"
    audio_path = "/mnt/hdd2/huanglingyu/vgg/datasets/Koala-36M-v1/videos/Koala_36M_3_sv/pv97fCP13SE_68/audio.wav"
    read_video_by_imageio(video_path, audio_path)
    # read_video_by_torchvision(video_path, audio_path)