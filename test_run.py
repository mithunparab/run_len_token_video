import torch
import torch.nn as nn
from torchvision.datasets import UCF101
from torchvision import transforms
from utils import visualize_moving_patches
from timesformer_model import TimeSformer, initialize_weights
import argparse

def test_timesformer_with_visualization(args):
    """
    Test the TimeSformer with the UCF101 dataset and visualize moving patches.
    """
    
    # Parameters (from argparse)
    ucf_data_dir = args.ucf_data_dir
    ucf_label_dir = args.ucf_label_dir
    frames_per_clip = args.frames_per_clip
    step_between_clips = args.step_between_clips
    batch_size = args.batch_size
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    patch_size = args.patch_size
    dim = args.dim
    depth = args.depth
    heads = args.heads
    dim_head = args.dim_head
    threshold = args.threshold
    max_run_length = args.max_run_length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformation pipeline
    tfs = transforms.Compose([
        transforms.Lambda(lambda x: x / 255.),  # Scale to [0, 1]
        transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # Reshape to (T, C, H, W)
        transforms.Lambda(lambda x: nn.functional.interpolate(x, size=(240, 240)))  # Resize to (240, 240)
    ])

    def custom_collate(batch):
        filtered_batch = []
        for video, _, label in batch:
            filtered_batch.append((video, label))
        return torch.utils.data.dataloader.default_collate(filtered_batch)

    # Create train and test loaders
    train_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip,
                           step_between_clips=step_between_clips, train=True, transform=tfs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=pin_memory,
                                               collate_fn=custom_collate)
    test_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip,
                          step_between_clips=step_between_clips, train=False, transform=tfs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, pin_memory=pin_memory,
                                              collate_fn=custom_collate)

    # Select a batch from the test loader
    try:
        video_tensor, label = next(iter(test_loader))
    except Exception as e:
        print("Error loading UCF101 dataset. Ensure that the dataset is downloaded correctly.")
        print(e)
        return

    video_tensor = video_tensor.to(device, non_blocking=True)  # Use non-blocking transfer
    print(f'Label: {label}')
    print(f'Video Tensor Shape: {video_tensor.shape}')  # Expected: [1, 5, 3, 240, 240]

    # Instantiate the TimeSformer model
    model = TimeSformer(
        dim=dim,
        num_frames=frames_per_clip,
        num_target_frames=frames_per_clip, 
        image_size=240,  
        patch_size=patch_size,
        channels=3, 
        out_channels=3,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        attn_dropout=0.1,
        ff_dropout=0.1,
        threshold=threshold,
        max_run_length=max_run_length
    ).to(device)

    # Initialize weights
    initialize_weights(model)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output, mask = model(video_tensor)  # Output: (B, num_target_frames, C, H, W), Mask: (B, F-1, n_patches)

    print(f"Input Video Shape: {video_tensor.shape}")  # [1, 5, 3, 240, 240]
    print(f"Output Frames Shape: {output.shape}")      # [1, 5, 3, 240, 240]
    print(f"Mask Shape: {mask.shape}")                # [1, 4, 225]

    # Visualize moving patches for the first video in the batch and first frame difference
    visualize_moving_patches(video_tensor, mask, patch_size, threshold, frame_idx=0)

    # Optionally visualize additional frames
    for f_idx in range(min(3, frames_per_clip - 1)):  # Visualize up to 3 frames
        visualize_moving_patches(video_tensor, mask, patch_size, threshold, frame_idx=f_idx)

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description="Test TimeSformer model with visualization on UCF101 dataset.")
    parser.add_argument("--ucf_data_dir", type=str, required=True, help="Path to the UCF101 video dataset.")
    parser.add_argument("--ucf_label_dir", type=str, required=True, help="Path to the UCF101 label dataset.")
    parser.add_argument("--frames_per_clip", type=int, default=5, help="Number of frames per clip.")
    parser.add_argument("--step_between_clips", type=int, default=1, help="Step between clips.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--pin_memory", type=bool, default=True, help="Pin memory for DataLoader.")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for TimeSformer.")
    parser.add_argument("--dim", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--depth", type=int, default=2, help="Number of layers in TimeSformer.")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--dim_head", type=int, default=64, help="Dimension of each attention head.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for visualization.")
    parser.add_argument("--max_run_length", type=int, default=5, help="Maximum run length for masking.")
    
    args = parser.parse_args()
    test_timesformer_with_visualization(args)