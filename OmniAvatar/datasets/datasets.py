import os
import warnings
import imageio
from PIL import Image
import pandas as pd
import json
import librosa
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor

# TODO 抄过来的，不要了，但有些东西可以参考
class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None,
        metadata_path=None,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        max_pixels=1920 * 1080,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat

        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True

        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[: -len(file_ext_name) - 1]
            if (
                file_ext_name not in self.image_file_extension
                and file_ext_name not in self.video_file_extension
            ):
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(
                os.path.join(folder, prompt_file_name), "r", encoding="utf-8"
            ) as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        image = torchvision.transforms.functional.center_crop(
            image, (target_height, target_width)
        )
        return image

    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while (
                num_frames > 1
                and num_frames % self.time_division_factor
                != self.time_division_remainder
            ):
                num_frames -= 1
        return num_frames

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames

    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames

    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension

    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension

    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None

    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data

    def __len__(self):
        return len(self.data) * self.repeat

# 调用这个dataset之前，要先调用数据集下面的那个generate_metadata的脚本，生成metadata.csv
class WanVideoDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        dataset_base_path = args.dataset_base_path
        metadata_path = os.path.join(dataset_base_path, "metadata.csv")
        self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.wav2vec_path
        )
        
        # 从预处理好的csv或者json里面读出来, 是个数组
        metadata = pd.read_csv(metadata_path)
        data_len = min(args.debug_data_len, len(metadata)) if args.debug else len(metadata)
        self.data = [metadata.iloc[i].to_dict() for i in range(data_len)]
    
    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        args = self.args
        
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
        origin_video_fps = info["video_fps"]  # TODO 这里应该就是用视频本身的fps吧？
        video_fps = int(round(origin_video_fps))
        video = video.float() / 255.0
        video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
        video = F.interpolate(video, size=(target_h, target_w), mode="bilinear", align_corners=False)
        video = video.permute(1, 0, 2, 3)  # [C, T, H, W]
        origin_video_len = video.shape[1]
        
        audio, sr = librosa.load(audio_path, sr=args.sample_rate)
        samples_per_frame = int(args.sample_rate / video_fps)

        # TODO 短于max的直接丢，不能扩展，扩展会让模型学错东西
        if origin_video_len <= max_frame:
            video_clip = video[:, :origin_video_len]
            audio_clip = audio[: origin_video_len * samples_per_frame]
            print(
                f"[OmniTrainingModule forward_preprocess] -> Video shorter than max_frame, use first {T} frames"
            )
        else:
            start_idx = np.random.randint(0, origin_video_len - max_frame + 1)
            video_clip = video[:, start_idx : start_idx + max_frame]
            audio_clip = audio[
                start_idx
                * samples_per_frame : (start_idx + max_frame)
                * samples_per_frame
            ]
            print(
                f"[OmniTrainingModule forward_preprocess] -> Video longer than max_frame, crop from {start_idx} to {start_idx + max_frame}"
            )
        L = video_clip.shape[1] # 这个L应该是=max_frame
        T = (L + 3) // 4
        
        # 音频特征提取
        audio_latent = np.squeeze(self.wav_feature_extractor(audio_clip, sampling_rate=args.sample_rate).input_values)
        audio_latent = torch.from_numpy(audio_latent)
        # audio_latent = audio_latent.unsqueeze(0)
        
        data['video'] = video_clip
        data['audio'] = audio_latent
        data['L'] = L
        data['T'] = T
        
        return data
    
    def __len__(self):
        return len(self.data)
    
# TODO validation数据集怎么拆？
class WanVideoValidationDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        dataset_base_path = args.dataset_base_path
        self.dataset_base_path = dataset_base_path
        metadata_path = os.path.join(dataset_base_path, "metadata.csv")
        self.metadata_path = metadata_path
        
        # 从预处理好的csv或者json里面读出来, 是个数组
        metadata = pd.read_csv(metadata_path)
        data_len = min(args.debug_data_len, len(metadata)) if args.debug else len(metadata)
        self.data = [metadata.iloc[i].to_dict() for i in range(data_len)]
    
    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        return data
    
    def __len__(self):
        return len(self.data)