import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Visualization Function
def visualize_moving_patches(video, mask, patch_size, threshold, frame_idx=None):
    """
    Visualize the moving patches in a video by reducing opacity of non-moving areas.

    Args:
        video (torch.Tensor): Input video tensor of shape (b, f, c, h, w).
        mask (torch.Tensor): Mask tensor indicating significant patches (b, f-1, n_patches).
        patch_size (int): Size of the patches.
        threshold (float): Threshold used for pruning.
        frame_idx (int, optional): Index of the frame to visualize. If None, visualize a random frame.
    """
    b, f, c, h, w = video.shape
    _, f_diff, n_patches = mask.shape
    p = patch_size

    if frame_idx is None:
        frame_idx = torch.randint(0, f_diff, (1,)).item()

    # Create a horizontal layout for the plots
    plt.figure(figsize=(15, 5))  # Adjust width for horizontal layout

    for b_idx in range(b):
        frame = video[b_idx, frame_idx].cpu().permute(1, 2, 0).numpy()  # Convert to (H, W, C) format
        mask_frame = mask[b_idx, frame_idx].cpu().numpy().reshape(h // p, w // p)

        # Create a mask image
        mask_img = np.kron(mask_frame, np.ones((p, p)))  # Scale up the patch mask

        # Reduce opacity of non-moving parts
        overlay = frame.copy()
        non_moving_factor = 0.3  # Factor to dim non-moving parts
        overlay[mask_img == 0] *= non_moving_factor  # Reduce intensity of non-moving regions

        # Plot in a horizontal grid
        plt.subplot(1, b, b_idx + 1)  # 1 row, `b` columns, current subplot
        plt.title(f'Batch {b_idx},  Moving Patches Highlighted  {frame_idx}')
        plt.imshow(np.clip(overlay, 0, 1))  # Ensure pixel values remain in range [0, 1]
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Weight Initialization
def initialize_weights(model):
    """
    Initialize model weights using Xavier initialization.

    Args:
        model (nn.Module): The model to initialize.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
