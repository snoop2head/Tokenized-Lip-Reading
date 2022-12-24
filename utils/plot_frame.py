import time
import matplotlib.pyplot as plt


def plot_frame(train_dataset, processor):
    sample_idx = 0
    (
        original_img,
        flipped_img,
        original_video,
        flipped_video,
        attention_mask,
        label,
    ) = train_dataset[sample_idx]

    print(original_video.size())  # C, T, H, W

    # check whether x channel is flipped
    if flipped_img == original_img:
        print("not flipped")
    else:
        print("flipped")

    for frame_idx in range(0, 29):
        print(f"frame_idx: {frame_idx}", "attend: ", attention_mask[frame_idx].item())
        print(f"label: {label}")
        print(f"label: {processor.tokenizer.decode(label)}")
        plt.imshow(flipped_video[0, frame_idx, ...], cmap="gray")
        plt.show()
        time.sleep(0.15)
