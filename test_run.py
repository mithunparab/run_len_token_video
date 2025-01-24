import torch
import torch.nn as nn
from torchvision.datasets import UCF101
from torchvision import transforms
from utils import visualize_moving_patches
from timesformer_model import TimeSformer, initialize_weights

def test_timesformer_with_visualization():
    """
    Test the TimeSformer with the UCF101 dataset and visualize moving patches.
    """
    
    # Parameters
    ucf_data_dir = "/kaggle/input/ucf101/UCF101/UCF-101"
    ucf_label_dir = "/kaggle/input/ucf101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist"
    frames_per_clip = 5
    step_between_clips = 1
    batch_size = 1  
    num_workers = 4 
    pin_memory = True  
    patch_size = 16
    dim = 512
    depth = 2  
    heads = 8
    dim_head = 64
    threshold = 0.5
    max_run_length = 5
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
# Run the Test Function with Visualization
if __name__ == "__main__":
    test_timesformer_with_visualization()
1