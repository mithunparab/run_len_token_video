import torch
from einops import rearrange

# Helper Functions
def compute_frame_differences(video):
    """
    Compute the differences between consecutive frames in a video.

    Args:
        video (torch.Tensor): Input video tensor of shape (b, f, c, h, w).

    Returns:
        torch.Tensor: Frame differences of shape (b, f-1, c, h, w).
    """
    frame_diffs = video[:, 1:] - video[:, :-1]
    return frame_diffs

def extract_patches(video, patch_size):
    """
    Extract patches from video frames.

    Args:
        video (torch.Tensor): Input tensor of shape (b, f, c, h, w).
        patch_size (int): Size of the patches.

    Returns:
        torch.Tensor: Patches of shape (b, f, num_patches, patch_dim).
    """
    p = patch_size
    patches = rearrange(video, 'b f c (h p1) (w p2) -> b f (h w) (p1 p2 c)', p1=p, p2=p)
    return patches

def compute_patch_differences(patches):
    """
    Compute magnitudes of patch differences.

    Args:
        patches (torch.Tensor): Patches tensor of shape (b, f-1, n_patches, dim).

    Returns:
        torch.Tensor: Magnitudes of patch differences of shape (b, f-1, n_patches).
    """
    patch_diff_magnitudes = torch.norm(patches, dim=-1)  # (b, f-1, n_patches)
    return patch_diff_magnitudes

def prune_patches(patch_diff_magnitudes, threshold):
    """
    Prune patches with low differences based on a threshold.

    Args:
        patch_diff_magnitudes (torch.Tensor): Magnitudes of patch differences.
        threshold (float): Threshold value to prune patches.

    Returns:
        torch.Tensor: Mask indicating significant patches.
    """
    mask = patch_diff_magnitudes > threshold  # Shape: (b, f-1, n_patches)
    return mask

def compute_temporal_run_lengths_vectorized(mask, max_run_length):
    """
    Compute temporal run-lengths for each patch in each frame using vectorized operations.

    Args:
        mask (torch.Tensor): Boolean tensor indicating significant patches (b, f-1, n_patches).
        max_run_length (int): Maximum run-length to cap the values.

    Returns:
        torch.Tensor: Run-lengths tensor of shape (b, f-1, n_patches).
    """
    # Invert mask: True where no change
    no_change = ~mask  # Shape: (b, f-1, n_patches)

    # Initialize run_lengths
    run_lengths = torch.zeros_like(no_change, dtype=torch.long)

    # Compute run-lengths using cumulative sums
    run_lengths[:, 0, :] = no_change[:, 0, :].long()
    for t in range(1, no_change.shape[1]):
        run_lengths[:, t, :] = torch.where(
            no_change[:, t, :],
            run_lengths[:, t-1, :] + 1,
            torch.zeros_like(run_lengths[:, t, :])
        )

    run_lengths = run_lengths.clamp(max=max_run_length)
    return run_lengths
