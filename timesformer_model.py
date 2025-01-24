import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum

from run_length_tokenizer_utils import compute_temporal_run_lengths_vectorized, compute_frame_differences, \
                                     extract_patches, compute_patch_differences, prune_patches

# Transformer Components
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn_scores = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn_scores, v)
    return out

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        dropout=0.,
        num_tokens=1,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.num_tokens = num_tokens

    def forward(self, x, einops_from, einops_to, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q = q * self.scale

        # Splitting out added target tokens
        cls_q, q_ = q[:, :self.num_tokens], q[:, self.num_tokens:]
        cls_k, k_ = k[:, :self.num_tokens], k[:, self.num_tokens:]
        cls_v, v_ = v[:, :self.num_tokens], v[:, self.num_tokens:]

        # Target tokens attend all the tokens in sequence
        cls_out = attn(cls_q, k, v)

        # Rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # Expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]

        cls_k, cls_v = map(lambda t: repeat(t, 'b t d -> (b r) t d', r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # Attention
        out = attn(q_, k_, v_)

        # Merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # Concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # Merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        # Combine heads out
        return self.to_out(out)

# TimeSformer Class
class TimeSformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        num_target_frames=4,
        image_size=240,
        patch_size=16,
        channels=3,
        out_channels=3,
        depth=12,
        heads=8,
        dim_head=64,
        attn_dropout=0.,
        ff_dropout=0.,
        threshold=0.1,  # Threshold for pruning patches
        max_run_length=10  # Maximum run-length for embeddings
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.dim = dim
        self.out_channels = out_channels
        self.num_frames = num_frames
        self.num_target_frames = num_target_frames

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2
        out_patch_dim = out_channels * patch_size ** 2

        self.num_tokens = num_target_frames * num_patches
        self.num_target_patches = self.num_tokens  # Each target frame has num_patches tokens

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.length_embedding = nn.Embedding(max_run_length + 1, dim)  # +1 to include zero run-length

        # Positional Embedding
        self.pos_emb = nn.Embedding(num_frames * num_patches + self.num_target_patches, dim)

        # Target Tokens
        self.target_tokens = nn.Parameter(torch.randn(self.num_target_patches, dim))

        # Transformer Layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # Time Attention, Spatial Attention, Feed Forward
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, num_tokens=self.num_target_patches)),
                PreNorm(dim, Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, num_tokens=self.num_target_patches)),
                PreNorm(dim, FeedForward(dim, dropout=ff_dropout))
            ]))

        # Output Projection
        self.to_dembedded_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_patch_dim)
        )

        self.threshold = threshold
        self.max_run_length = max_run_length

    def forward(self, video):
        """
        Forward pass of the TimeSformer.

        Args:
            video (torch.Tensor): Input video tensor of shape (b, f, c, h, w).

        Returns:
            torch.Tensor: Predicted future frames of shape (b, num_target_frames, out_channels, h, w).
            torch.Tensor: Mask indicating significant patches.
        """
        b, f, c, h, w = video.shape
        device = video.device
        p = self.patch_size

        assert h % p == 0 and w % p == 0, f'Height {h} and width {w} must be divisible by the patch size {p}.'

        n_patches = (h // p) * (w // p)

        # Compute frame differences
        frame_diffs = compute_frame_differences(video)  # Shape: (b, f-1, c, h, w)

        # Extract patches from frame differences
        diff_patches = extract_patches(frame_diffs, p)  # Shape: (b, f-1, n_patches, dim)

        # Compute patch differences magnitudes
        patch_diff_magnitudes = compute_patch_differences(diff_patches)  # (b, f-1, n_patches)

        # Prune patches with low differences
        mask = prune_patches(patch_diff_magnitudes, self.threshold)  # (b, f-1, n_patches)

        # Ensure that at least one patch is selected per frame to avoid all patches being pruned
        frame_masks = mask.any(dim=-1, keepdim=True)  # (b, f-1, 1)
        if not frame_masks.all():
            # Find frames with no significant patches
            no_significant = ~frame_masks.squeeze(-1)
            for b_idx, f_idx in zip(*no_significant.nonzero(as_tuple=True)):
                # Select the patch with the highest magnitude in this frame
                _, top_patch = patch_diff_magnitudes[b_idx, f_idx].max(dim=0)
                mask[b_idx, f_idx, top_patch] = True

        # Compute temporal run-lengths (vectorized)
        run_lengths = compute_temporal_run_lengths_vectorized(~mask, self.max_run_length)  # Shape: (b, f-1, n_patches)

        # Extract patches from the original video frames (excluding the last frame)
        patches = extract_patches(video[:, :-1], p)  # Shape: (b, f-1, n_patches, dim)

        # Apply mask to keep only significant patches
        mask_expanded = mask.unsqueeze(-1).expand_as(patches)  # (b, f-1, n_patches, dim)
        significant_patches = patches * mask_expanded.float()  # Zero out insignificant patches

        # Flatten batch and frames
        significant_patches = rearrange(significant_patches, 'b f n d -> (b f n) d')  # (b*f*n, d)

        # Embed patches
        tokens = self.to_patch_embedding(significant_patches)  # (b*f*n, dim)

        # Add length embeddings
        run_lengths = rearrange(run_lengths, 'b f n -> (b f n)')  # (b*f*n,)
        length_embeds = self.length_embedding(run_lengths)  # (b*f*n, dim)
        tokens += length_embeds  # (b*f*n, dim)

        # Reshape tokens back to (b, f*n, dim)
        tokens = rearrange(tokens, '(b f n) d -> b f n d', b=b, f=f-1, n=n_patches)  # (b, f-1, n, d)

        # Concatenate target tokens
        target_tokens = repeat(self.target_tokens, 'n d -> b n d', b=b)  # (b, num_target_patches, dim)
        x = torch.cat((target_tokens, tokens.view(b, -1, self.dim)), dim=1)  # (b, num_target_patches + f*n, dim)

        # print(f'x shape after concatenation: {x.shape}')  # Debugging

        # Add positional embeddings
        pos_indices = torch.arange(x.shape[1], device=device).unsqueeze(0).expand(b, -1)  # (b, seq_len)
        x += self.pos_emb(pos_indices)  # Broadcasting addition

        # Pass through Transformer layers
        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n=n_patches) + x  # Time Attention
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f=f-1) + x      # Spatial Attention
            x = ff(x) + x                                             # Feed Forward

        # Extract output tokens (only target tokens)
        out_tokens = x[:, :self.num_target_patches]  # (b, num_target_patches, dim)
        # print(f'out_tokens shape before to_dembedded_out: {out_tokens.shape}')  # Debugging

        # Embed to output patch dimension
        out_tokens = self.to_dembedded_out(out_tokens)  # (b, num_target_patches, out_patch_dim)
        # print(f'out_tokens shape after to_dembedded_out: {out_tokens.shape}')  # Debugging

        # Reshape to (b, f, h_p, w_p, c, p1, p2)
        out_tokens = rearrange(
            out_tokens,
            'b (f h_p w_p) (c p1 p2) -> b f h_p w_p c p1 p2',
            f=self.num_target_frames,
            h_p=h // self.patch_size,
            w_p=w // self.patch_size,
            c=self.out_channels,
            p1=self.patch_size,
            p2=self.patch_size
        )
        # print(f'out_tokens shape after rearrange step 1: {out_tokens.shape}')  # Debugging

        # Merge spatial dimensions
        out_frames = rearrange(
            out_tokens,
            'b f h_p w_p c p1 p2 -> b f c (h_p p1) (w_p p2)'
        )
        # print(f'out_frames shape after rearrange step 2: {out_frames.shape}')  # Debugging

        return out_frames, mask