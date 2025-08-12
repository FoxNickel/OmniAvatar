import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from ..utils.io_utils import hash_state_dict_keys
from .audio_pack import AudioPack
from ..utils.args_config import args
from xfuser.core.distributed import (get_sequence_parallel_rank,
                                     get_sequence_parallel_world_size,
                                     get_sp_group)
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

# DiTBlock 是 WanModel 的核心构建单元，结合自注意力、跨模态注意力、MLP、归一化和时序调制，实现时空建模和多模态条件融合，是视频生成 Transformer 的关键模块。
class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        # 多头自注意力层
        self.self_attn = SelfAttention(dim, num_heads, eps)
        # 多头跨模态注意力层（支持图片条件）
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        # 三个LayerNorm归一化层
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        # 前馈网络（MLP）
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        # 残差门控模块
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        # t_mod: 当前时间步的调制参数，和modulation参数相加后分成6份
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))

        # 2. 跨模态注意力分支（如文本/图片条件）
        x = x + self.cross_attn(self.norm3(x), context)

        # 3. 前馈网络分支
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x



class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,                # 主干特征维度（Transformer内部通道数），决定模型容量和表达能力
        in_dim: int,             # 输入特征维度（如视频latent的通道数），用于patch embedding的输入
        ffn_dim: int,            # 前馈网络（MLP）隐藏层维度，影响每个block的非线性建模能力
        out_dim: int,            # 输出特征维度（如视频latent的通道数），用于还原输出patch
        text_dim: int,           # 文本embedding的输入维度（如CLIP/TextEncoder输出的维度）
        freq_dim: int,           # 时间步嵌入的输入维度，用于扩散时间步的编码
        eps: float,              # LayerNorm/RMSNorm等归一化的数值稳定参数
        patch_size: Tuple[int, int, int], # patch的三维尺寸（帧数, 高度, 宽度），用于视频分块
        num_heads: int,          # Transformer自注意力的头数，影响时空建模能力
        num_layers: int,         # Transformer block的层数，决定模型深度
        has_image_input: bool,   # 是否支持图片条件输入（如i2v任务），决定是否启用图片相关模块
        audio_hidden_size: int=32, # 音频条件隐藏层维度，影响音频特征的表达能力
    ):
        super().__init__()
        # Using WanModel with dim=1536, in_dim=33, ffn_dim=8960, out_dim=16, text_dim=4096, freq_dim=256, eps=1e-06, patch_size=[1, 2, 2], num_heads=12, num_layers=30, has_image_input=False, audio_hidden_size=32
        print(f"Using WanModel with dim={dim}, in_dim={in_dim}, ffn_dim={ffn_dim}, out_dim={out_dim}, text_dim={text_dim}, freq_dim={freq_dim}, eps={eps}, patch_size={patch_size}, num_heads={num_heads}, num_layers={num_layers}, has_image_input={has_image_input}, audio_hidden_size={audio_hidden_size}")

        # 保存主要参数为成员变量
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        # Patch Embedding：用3D卷积将输入视频分块并升维到dim
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
            # nn.LayerNorm(dim) # 可选归一化

        # 文本嵌入层：将文本embedding投影到主干维度
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )

        # 时间步嵌入层：将扩散时间步编码为向量
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        # 时间步调制层：进一步投影为6倍维度，用于调制block内部行为
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # Transformer Blocks：堆叠多个DiTBlock，每个block负责时空/条件建模
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])

        # 输出头部：将主干特征还原为输出patch
        self.head = Head(dim, out_dim, patch_size, eps)

        # RoPE位置编码：预计算3D旋转位置编码频率，用于时空自注意力
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        # 图片输入支持：如果启用，定义MLP将CLIP图像特征投影到主干维度
        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280

        # 音频输入支持：根据配置决定是否启用音频条件
        if 'use_audio' in args:
            self.use_audio = args.use_audio
        else:
            self.use_audio = False
        if self.use_audio:
            audio_input_dim = 10752
            audio_out_dim = dim
            # 音频特征投影模块
            self.audio_proj = AudioPack(audio_input_dim, [4, 1, 1], audio_hidden_size, layernorm=True)
            # 为部分block准备音频条件投影层
            self.audio_cond_projs = nn.ModuleList()
            for d in range(num_layers // 2 - 1):
                l = nn.Linear(audio_hidden_size, audio_out_dim)
                self.audio_cond_projs.append(l)
        
        # 打印各主要模块结构和参数
        print("\n[WanModel] patch_embedding:", self.patch_embedding)
        print("[WanModel] text_embedding:", self.text_embedding)
        print("[WanModel] time_embedding:", self.time_embedding)
        print("[WanModel] time_projection:", self.time_projection)
        print("[WanModel] blocks (DiTBlock):")
        for i, block in enumerate(self.blocks):
            print(f"  Block {i}: {block}")
        print("[WanModel] head:", self.head)
        print("[WanModel] RoPE freqs shape:", [f.shape for f in self.freqs])
        if hasattr(self, "img_emb"):
            print("[WanModel] img_emb:", self.img_emb)
        if hasattr(self, "audio_proj"):
            print("[WanModel] audio_proj:", self.audio_proj)
        if hasattr(self, "audio_cond_projs"):
            print("[WanModel] audio_cond_projs:")
            for i, proj in enumerate(self.audio_cond_projs):
                print(f"  Audio Cond Proj {i}: {proj}")

    def patchify(self, x: torch.Tensor):
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
            x: torch.Tensor,                      # 输入视频latent或特征
            timestep: torch.Tensor,               # 当前扩散时间步（整数或向量）
            context: torch.Tensor,                # 文本条件embedding
            clip_feature: Optional[torch.Tensor] = None, # 可选CLIP图像特征（未用到）
            y: Optional[torch.Tensor] = None,     # 图片条件embedding（如mask拼接）
            use_gradient_checkpointing: bool = False,     # 是否使用梯度检查点，节省显存
            audio_emb: Optional[torch.Tensor] = None,     # 音频条件embedding
            use_gradient_checkpointing_offload: bool = False, # 是否将checkpoint临时存CPU
            tea_cache = None,                     # 分布式推理缓存
            **kwargs,
            ):
        print(f"[WanModel] Forward pass with x shape: {x.shape}, timestep: {timestep.shape}, context shape: {context.shape}, y shape: {y.shape if y is not None else 'None'}, audio_emb shape: {audio_emb.shape if audio_emb is not None else 'None'}")
        # 1. 时间步嵌入与调制参数
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        
        # 2. 文本条件嵌入
        context = self.text_embedding(context)
        
        # 3. 获取输入空间尺寸
        lat_h, lat_w = x.shape[-2], x.shape[-1]

        # 4. 音频条件处理（如果启用）
        if audio_emb != None and self.use_audio: # TODO  cache
            audio_emb = audio_emb.permute(0, 2, 1)[:, :, :, None, None] # 调整维度
            audio_emb = torch.cat([audio_emb[:, :, :1].repeat(1, 1, 3, 1, 1), audio_emb], 2) # 把第三维的第一个拿出来，repeat 3次
            audio_emb = self.audio_proj(audio_emb) # 投影到隐藏层

            audio_emb = torch.concat([audio_cond_proj(audio_emb) for audio_cond_proj in self.audio_cond_projs], 0)  # 多层融合

        # 5. 拼接图片条件（如mask），做patch embedding
        # 训练时，x 是视频的 latent，y 是图片条件（如图片 latent + mask），拼接后送入 patch embedding 和 patchify
        x = torch.cat([x, y], dim=1)  # 拼接图片条件
        x = self.patch_embedding(x)   # 3D卷积分块升维
        x, (f, h, w) = self.patchify(x)  # 展平成patch序列

        # 6. 计算RoPE三维位置编码
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)  # (patch数, 1, 维度)

        # 7. 分布式推理缓存检查
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        if tea_cache is not None:
            tea_cache_update = tea_cache.check(self, x, t_mod)
        else:
            tea_cache_update = False
        ori_x_len = x.shape[1]
        if tea_cache_update:
            x = tea_cache.update(x)
        else:
            # 8. 分布式并行支持（如多卡推理）
            if args.sp_size > 1:
                # Context Parallel
                sp_size = get_sequence_parallel_world_size()
                pad_size = 0
                if ori_x_len % sp_size != 0:
                    pad_size = sp_size - ori_x_len % sp_size
                    x = torch.cat([x, torch.zeros_like(x[:, -1:]).repeat(1, pad_size, 1)], 1)
                x = torch.chunk(x, sp_size, dim=1)[get_sequence_parallel_rank()]

            # 9. 音频条件reshape，适配block输入
            audio_emb = audio_emb.reshape(x.shape[0], audio_emb.shape[0] // x.shape[0], -1, *audio_emb.shape[2:])
            
            # 10. 主干Transformer block循环
            for layer_i, block in enumerate(self.blocks):
                # audio cond
                # 音频条件融合（部分block）
                if self.use_audio:
                    au_idx = None
                    if (layer_i <= len(self.blocks) // 2 and layer_i > 1): # < len(self.blocks) - 1:
                        au_idx = layer_i - 2
                        audio_emb_tmp = audio_emb[:, au_idx].repeat(1, 1, lat_h // 2, lat_w // 2, 1) # 1, 11, 45, 25, 128
                        audio_cond_tmp = self.patchify(audio_emb_tmp.permute(0, 4, 1, 2, 3))[0]
                        if args.sp_size > 1:
                            if pad_size > 0:    
                                audio_cond_tmp = torch.cat([audio_cond_tmp, torch.zeros_like(audio_cond_tmp[:, -1:]).repeat(1, pad_size, 1)], 1)
                            audio_cond_tmp = torch.chunk(audio_cond_tmp, sp_size, dim=1)[get_sequence_parallel_rank()]
                        x = audio_cond_tmp + x # 音频条件加到主干特征

                # 11. 梯度检查点（节省显存，训练时用）
                if self.training and use_gradient_checkpointing:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block),
                                x, context, t_mod, freqs,
                                use_reentrant=False,
                            )
                    else:
                        # TODO 训练的时候 checkpoint要换，checkpoint的作用是减少显存消耗
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = block(x, context, t_mod, freqs)
            # 12. 分布式缓存同步
            if tea_cache is not None:
                x_cache = get_sp_group().all_gather(x, dim=1) # TODO: the size should be devided by sp_size
                x_cache = x_cache[:, :ori_x_len]
                tea_cache.store(x_cache)

        # 13. 输出头部还原patch
        x = self.head(x, t)
        if args.sp_size > 1:
            # Context Parallel
            x = get_sp_group().all_gather(x, dim=1) # TODO: the size should be devided by sp_size
            x = x[:, :ori_x_len]

        # 14. unpatchify还原为视频帧格式
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        else:
            config = {}
        if hasattr(args, "model_config"):
            model_config = args.model_config
            if model_config is not None:
                config.update(model_config)        
        return state_dict, config
