import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_real_path(target_path: str) -> str:
    """Path ko automatic fix karta hai agar VS Code kisi aur folder se run kare."""
    if os.path.exists(target_path):
        return target_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path = os.path.join(script_dir, target_path)
    if os.path.exists(alt_path):
        return alt_path
    return target_path

def inspect_image(image_path: str) -> dict:
    resolved_path = get_real_path(image_path)
    source_bgr = cv2.imread(resolved_path)
    if source_bgr is None:
        raise FileNotFoundError(f"Target image missing at location: {image_path}")
    
    h_dim, w_dim, ch_count = source_bgr.shape
    total_px = h_dim * w_dim
    est_memory = source_bgr.nbytes
    
    return {
        "width": int(w_dim),
        "height": int(h_dim),
        "channels": int(ch_count),
        "shape": [int(h_dim), int(w_dim), int(ch_count)],
        "pixel_count": int(total_px),
        "estimated_bytes": int(est_memory),
        "color_order": "BGR"
    }

def create_pixel_views(image_path: str, output_dir: str) -> dict:
    resolved_path = get_real_path(image_path)
    os.makedirs(output_dir, exist_ok=True)
    raw_img = cv2.imread(resolved_path)
    if raw_img is None:
        raise FileNotFoundError(f"Target image missing at location: {image_path}")
        
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    ch_b, ch_g, ch_r = cv2.split(rgb_img)
    gray_view = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    
    orig_h, orig_w = raw_img.shape[:2]
    scaled_w, scaled_h = orig_w // 2, orig_h // 2
    downscaled_img = cv2.resize(rgb_img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    
    fig = plt.figure(figsize=(13, 7))
    plots = [
        (1, ch_r, 'Red Channel View', 'Reds'),
        (2, ch_g, 'Green Channel View', 'Greens'),
        (3, ch_b, 'Blue Channel View', 'Blues'),
        (4, gray_view, 'Grayscale View', 'gray'),
        (5, downscaled_img, 'Half-Resolution View', None)
    ]
    
    for idx, data, title, cmap in plots:
        ax = fig.add_subplot(2, 3, idx)
        if cmap:
            ax.imshow(data, cmap=cmap)
        else:
            ax.imshow(data)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        
    fig.add_subplot(2, 3, 6).axis('off')
    plt.tight_layout()
    file_target = os.path.join(output_dir, "pixel_views.png")
    plt.savefig(file_target, dpi=150)
    plt.close(fig)
    
    return {
        "original_size": [orig_w, orig_h],
        "downsampled_size": [scaled_w, scaled_h],
        "output_path": file_target
    }

def create_adjustments(
    image_path: str,
    output_dir: str,
    brightness_delta: int = 40,
    contrast_factor: float = 1.5,
    threshold: int = 127,
) -> dict:
    resolved_path = get_real_path(image_path)
    os.makedirs(output_dir, exist_ok=True)
    base_bgr = cv2.imread(resolved_path)
    if base_bgr is None:
        raise FileNotFoundError(f"Target image missing at location: {image_path}")
        
    gray_base = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    bright_map = cv2.add(gray_base, np.array([float(brightness_delta)]))
    contrast_map = cv2.convertScaleAbs(gray_base, alpha=contrast_factor, beta=0)
    thresh_flag, binary_map = cv2.threshold(gray_base, threshold, 255, cv2.THRESH_BINARY)
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(bright_map, cmap='gray')
    axes[0].set_title(f'Brightness Adjustment (+{brightness_delta})', fontsize=10)
    axes[1].imshow(contrast_map, cmap='gray')
    axes[1].set_title(f'Contrast Scaling (x{contrast_factor})', fontsize=10)
    axes[2].imshow(binary_map, cmap='gray')
    axes[2].set_title(f'Binary Thresholding ({threshold})', fontsize=10)
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    file_target = os.path.join(output_dir, "adjustments.png")
    plt.savefig(file_target, dpi=150)
    plt.close(fig)
    
    return {
        "brightness_delta": brightness_delta,
        "contrast_factor": contrast_factor,
        "threshold": threshold,
        "output_path": file_target
    }

def create_blur_and_edges(
    image_path: str,
    output_dir: str,
    kernel_size: int = 5,
) -> dict:
    resolved_path = get_real_path(image_path)
    os.makedirs(output_dir, exist_ok=True)
    src_bgr = cv2.imread(resolved_path)
    if src_bgr is None:
        raise FileNotFoundError(f"Target image missing at location: {image_path}")
        
    gray_src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.blur(gray_src, (kernel_size, kernel_size))
    
    gx_raw = cv2.Sobel(gray_src, cv2.CV_64F, 1, 0, ksize=3)
    gy_raw = cv2.Sobel(gray_src, cv2.CV_64F, 0, 1, ksize=3)
    mag_raw = np.hypot(gx_raw, gy_raw)
    
    gx_blur = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    gy_blur = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    mag_blur = np.hypot(gx_blur, gy_blur)
    
    fig, grid_axes = plt.subplots(2, 2, figsize=(11, 8.5))
    grid_axes[0, 0].imshow(gray_src, cmap='gray')
    grid_axes[0, 0].set_title('Original Gray Input')
    grid_axes[0, 1].imshow(smoothed, cmap='gray')
    grid_axes[0, 1].set_title(f'Mean Filtered ({kernel_size}x{kernel_size})')
    grid_axes[1, 0].imshow(mag_raw, cmap='gray')
    grid_axes[1, 0].set_title('Sobel Magnitude (Unfiltered)')
    grid_axes[1, 1].imshow(mag_blur, cmap='gray')
    grid_axes[1, 1].set_title('Sobel Magnitude (Smoothed)')
    
    for ax in grid_axes.flat:
        ax.axis('off')
        
    plt.tight_layout()
    file_target = os.path.join(output_dir, "blur_and_edges.png")
    plt.savefig(file_target, dpi=150)
    plt.close(fig)
    
    return {
        "kernel_size": kernel_size,
        "output_path": file_target
    }

def run_lab(image_path: str, output_dir: str) -> dict:
    t1 = inspect_image(image_path)
    t2 = create_pixel_views(image_path, output_dir)
    t3 = create_adjustments(image_path, output_dir)
    t4 = create_blur_and_edges(image_path, output_dir)
    
    return {
        "task1": t1,
        "task2": t2,
        "task3": t3,
        "task4": t4
    }

def main() -> None:
    input_file = "images/original.jpg"
    target_folder = "outputs"
    os.makedirs(target_folder, exist_ok=True)
    run_lab(input_file, target_folder)

if __name__ == "__main__":
    main()