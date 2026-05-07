import os
import subprocess
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt


DCRAW_EXE = "./my_stego_dcraw" 

def develop_images(image_name, base_pair, target_pair, image_path):
    clean_cmd = [DCRAW_EXE, "-c", "-T", "-4", "-W", "-q", "3", image_path]

    clean_image = f"clean_final_{image_name}.tiff"
    with open(clean_image, "wb") as f:
        subprocess.run(clean_cmd, stdout=f, stderr=subprocess.DEVNULL)
    stego_cmd = [DCRAW_EXE, "-c", "-Y", base_pair[0],base_pair[1], target_pair[0], target_pair[1], 
                 "-T", "-4", "-W", "-q", "3", image_path]
    
    injected_image = f"injected_final_b_{base_pair}_t_{target_pair}_{image_name}.tiff"
    with open(injected_image, "wb") as f:
        subprocess.run(stego_cmd, stdout=f, stderr=subprocess.DEVNULL)
    return clean_image, injected_image

def analyze_results(clean_image, injected_image,image_name, base_pair, target_pair):
    img_clean = iio.imread(clean_image).astype(np.float32)
    img_stego = iio.imread(injected_image).astype(np.float32)
    diff = np.abs(img_clean - img_stego)
    diff_map = np.mean(diff, axis=-1)

    plt.figure(figsize=(12, 8))
    plt.imshow(diff_map, cmap='binary', vmin=0, vmax=np.percentile(diff_map, 99.5))
    plt.colorbar(label='Absolute Pixel Difference (16-bit)')
    plt.title(f"Noise Heatmap for {image_name}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"data/post_demosaic_heatmap_b_{base_pair}_t_{target_pair}_{image_name}.png", dpi=300, bbox_inches="tight", facecolor='black')

if __name__ == "__main__":
    image_paths = []
    base_value_pairs = [(9.88e-08, 2.61e-09)]
    target_value_pairs = [(0.00059563, 6.8e-07)]
    if len(image_paths)!=len(base_value_pairs) or len(image_paths) != len(target_value_pairs):
        raise Exception("List do not have the same length")
    
    for index in len(image_paths):
        image_name = image_paths[index].split("/")[-1].split(".")[0]

        clean_image, injected_image = develop_images(image_name, base_value_pairs[index], target_value_pairs[index], image_paths[index])
        analyze_results( clean_image, injected_image, image_name, base_value_pairs[index], target_value_pairs[index])