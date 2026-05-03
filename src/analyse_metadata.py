from random import seed
import pandas as pd
import rawpy
import numpy as np
import exifread
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt
import imageio


def read_metadata(json_path):
    """
    Reads the ALSKA metadata from a JSON file and returns it as a pandas DataFrame.

    Parameters:
    json_path (str): The path to the JSON file containing the metadata.

    Returns:
    pd.DataFrame: A DataFrame containing the metadata.
    """
    try:
        df = pd.read_json(json_path)
        print("\nAvailable Columns:")
        print(df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None


def extract_base_iso(image_path):
    """
    Extracts the physical Base ISO speed rating directly from the DNG EXIF hardware tags.
    """
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
            iso_tag = tags.get("EXIF ISOSpeedRatings")

            if iso_tag:
                return int(iso_tag.values[0])
            else:
                print(f"Warning: No ISO tag found in {image_path}")
                return None
    except Exception as e:
        print(f"Error reading EXIF data: {e}")
        return None


def check_cfa_pattern(image_path):
    """
    Reads the physical hardware Color Filter Array layout directly from the RAW file.
    """
    with rawpy.imread(image_path) as raw:
        color_desc = raw.color_desc.decode("utf-8")
        pattern = raw.raw_pattern

        print(f"--- Sensor Diagnostics for {image_path.split('/')[-1]} ---")
        print(f"Hardware Color Map: {color_desc}")
        print(f"Hardware Matrix Pattern:\n{pattern}")

        cfa_layout = ""
        for row in pattern:
            for col in row:
                cfa_layout += color_desc[col]
            cfa_layout += "\n"

        print(f"Physical 2x2 Pixel Layout:{cfa_layout}")


def extract_raw_image(image_path):
    """
    Dynamically linearizes and de-interleaves the raw image based on the
    exact hardware sensor pattern (Bayer phase) embedded in the file.
    """
    try:
        with rawpy.imread(image_path) as raw:
            cfa_array = raw.raw_image_visible.astype(np.float32)
            color_desc = raw.color_desc.decode("utf-8")
            pattern = raw.raw_pattern
            bl = raw.black_level_per_channel

            bl_matrix = np.zeros_like(cfa_array)
            channels = {}
            g_count = 1

            for row in range(2):
                for col in range(2):
                    color_index = pattern[row, col]
                    color_letter = color_desc[color_index]

                    bl_matrix[row::2, col::2] = bl[color_index]

                    slice_view = (slice(row, None, 2), slice(col, None, 2))

                    if color_letter == "R":
                        channels["R"] = slice_view
                    elif color_letter == "B":
                        channels["B"] = slice_view
                    elif color_letter == "G":
                        channels[f"G{g_count}"] = slice_view
                        g_count += 1

            cfa_linear = np.maximum(cfa_array - bl_matrix, 0)

            R = cfa_linear[channels["R"]]
            G1 = cfa_linear[channels["G1"]]
            G2 = cfa_linear[channels["G2"]]
            B = cfa_linear[channels["B"]]

            return R, G1, G2, B, cfa_linear, cfa_linear.shape, channels

    except Exception as e:
        print(f"Error processing raw image: {e}")
        return None, None, None, None, None, None


def estimate_base_signal(R, G1, G2, B, original_shape, max_val, channel_slices):
    """
    Applies BayesShrink wavelet denoising to each color channel independently
    to estimate the clean base signal (μ), then dynamically re-interleaves them
    using the exact hardware sensor pattern.
    """
    norm_factor = float(max_val)

    def denoise_channel(channel):
        norm_channel = channel / norm_factor
        denoised_norm = denoise_wavelet(
            norm_channel,
            method="BayesShrink",
            mode="soft",
            wavelet_levels=3,
            wavelet="db2",
            rescale_sigma=True,
        )
        return denoised_norm * norm_factor

    R_mu = denoise_channel(R)
    G1_mu = denoise_channel(G1)
    G2_mu = denoise_channel(G2)
    B_mu = denoise_channel(B)

    mu_matrix = np.zeros(original_shape, dtype=np.float32)

    mu_matrix[channel_slices["R"]] = R_mu
    mu_matrix[channel_slices["G1"]] = G1_mu
    mu_matrix[channel_slices["G2"]] = G2_mu
    mu_matrix[channel_slices["B"]] = B_mu

    return mu_matrix


def benchmark(
    original_cfa,
    mu_matrix,
    a_target,
    b_target,
    image_name,
    sensor_max=65535.0,
    seed=93,
):
    """
    Benchmarks the absolute measured variance of the original and stego images
    against the theoretical physical models.
    """
    np.random.seed(seed)

    original_noise = original_cfa - mu_matrix
    mu_flat = mu_matrix.flatten()
    orig_noise_flat = original_noise.flatten()

    bins = np.linspace(0, np.max(mu_flat), 50)
    bin_indices = np.digitize(mu_flat, bins)

    meas_orig_var = []
    bin_centers = []

    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) > 500:
            meas_orig_var.append(np.var(orig_noise_flat[mask]))
            bin_centers.append((bins[i - 1] + bins[i]) / 2)

    a_base, b_base = estimate_sensor_parameters(bin_centers, meas_orig_var, sensor_max)

    mu_norm = mu_matrix / sensor_max
    var_delta_norm = (a_target * mu_norm + b_target) - (a_base * mu_norm + b_base)
    var_delta_norm = np.maximum(var_delta_norm, 0)

    injected_noise = (np.sqrt(var_delta_norm) * sensor_max) * np.random.randn(
        *mu_matrix.shape
    )

    stego_noise_total = (original_cfa + injected_noise) - mu_matrix
    stego_noise_flat = stego_noise_total.flatten()

    meas_stego_var = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) > 500:
            meas_stego_var.append(np.var(stego_noise_flat[mask]))

    bin_centers_norm = np.array(bin_centers) / sensor_max
    theoretical_base = (a_base * bin_centers_norm + b_base) * (sensor_max**2)
    theoretical_target = (a_target * bin_centers_norm + b_target) * (sensor_max**2)

    plt.figure(figsize=(12, 7))

    plt.scatter(
        bin_centers,
        meas_orig_var,
        alpha=0.6,
        color="green",
        label="Measured Original Variance",
    )
    plt.scatter(
        bin_centers,
        meas_stego_var,
        alpha=0.8,
        color="blue",
        label="Measured Stego Variance",
    )

    plt.plot(
        bin_centers,
        theoretical_base,
        color="lightgreen",
        linewidth=2,
        linestyle="--",
        label="Theoretical Base Profile",
    )
    plt.plot(
        bin_centers,
        theoretical_target,
        color="red",
        linewidth=2,
        label="Theoretical Target Profile",
    )

    plt.title(
        f"Cover-Source Switching Absolute Benchmark\nBase({a_base:.2e}, {b_base:.2e}) -> Target({a_target}, {b_target})"
    )
    plt.xlabel("Pixel Brightness (\u03bc)")
    plt.ylabel("Absolute Noise Variance (\u03c3\u00b2)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(
        f"data/plots/benchmark_plot_{image_name}.png", dpi=300, bbox_inches="tight"
    )

    return a_base, b_base


def inject_and_export_tiff(
    image_path,
    mu_matrix,
    a_base,
    b_base,
    a_target,
    b_target,
    image_name,
    sensor_max=65535.0,
    seed=93,
):
    """
    Injects the mathematical noise payload into the original raw sensor data
    using normalized physical parameters.
    """
    output_filename = f"data/stego_output/stego_output_{image_name}.tiff"
    mu_norm = mu_matrix / sensor_max

    var_delta_norm = (a_target * mu_norm + b_target) - (a_base * mu_norm + b_base)
    var_delta_norm = np.maximum(var_delta_norm, 0)

    np.random.seed(seed)
    stego_noise = (np.sqrt(var_delta_norm) * sensor_max) * np.random.randn(
        *mu_matrix.shape
    )

    with rawpy.imread(image_path) as raw:
        original_cfa = raw.raw_image_visible.astype(np.float32)
        poisoned_cfa = original_cfa + stego_noise

        # 16 bit
        poisoned_cfa = np.clip(poisoned_cfa, 0, 65535).astype(np.uint16)
        raw.raw_image_visible[:, :] = poisoned_cfa
        rgb_image = raw.postprocess(use_camera_wb=True, output_bps=16)

    imageio.imsave(output_filename, rgb_image)


def estimate_sensor_parameters(bin_centers, measured_variances, sensor_max=65535.0):
    """
    Estimates the physical camera sensor parameters (a_base, b_base)
    from the statistical variance of a clean, natural image.

    Parameters:
    - bin_centers: List or array of the pixel brightness bins (μ).
    - measured_variances: List or array of the absolute measured variance (σ²) per bin.
    - sensor_max: The physical saturation limit of the sensor (e.g., 65535 for 16-bit).

    Returns:
    - a_estimated: The estimated shot noise parameter (slope).
    - b_estimated: The estimated read noise parameter (y-intercept).
    """
    mu_norm_centers = np.array(bin_centers) / sensor_max
    var_norm_measured = np.array(measured_variances) / (sensor_max**2)

    a_estimated, b_estimated = np.polyfit(mu_norm_centers, var_norm_measured, 1)

    print(f"Estimated a_base (Shot Noise): {a_estimated:.4e}")
    print(f"Estimated b_base (Read Noise): {b_estimated:.4e}")

    return a_estimated, b_estimated


def export_original_image_tiff(image_path, image_name):
    """
    Exports the original raw image as a TIFF for visual comparison.
    """
    output_filename = f"data/original_tiff/original_{image_name}.tiff"
    with rawpy.imread(image_path) as raw:
        rgb_image = raw.postprocess(use_camera_wb=True, output_bps=16)
    imageio.imsave(output_filename, rgb_image)


def analyze_stego_difference(orig_path, stego_path, image_name):
    """
    Analyzes the mathematical and visual difference between the clean TIFF and the Stego TIFF.
    """

    img_orig = imageio.v2.imread(orig_path).astype(np.float32)
    img_stego = imageio.v2.imread(stego_path).astype(np.float32)

    diff = np.abs(img_orig - img_stego)

    diff_map = np.mean(diff, axis=-1)
    plt.figure(figsize=(12, 8))

    plt.imshow(diff_map, cmap="inferno", vmin=0, vmax=np.percentile(diff_map, 99))

    plt.colorbar(label="Absolute Pixel Difference (16-bit scale)")
    plt.title(f"Steganographic Noise Heatmap")
    plt.axis("off")
    plt.tight_layout()

    save_path = f"data/heatmap/stego_heatmap_{image_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="black")


def main():
    image_path = "data/Conversion_script/devHome/ALASKA_v2_RAWs/00001.dng"
    image_name = image_path.split("/")[-1].split(".")[0]
    # metadata_path = "data/alaska_metadata.json"
    # metadata_df = read_metadata(metadata_path)
    # iso_value = extract_base_iso(image_path)
    # print(f"Extracted Base ISO: {iso_value}")

    R, G1, G2, B, cfa_linear, original_shape, channels = extract_raw_image(image_path)
    max_val = max(np.max(R), np.max(G1), np.max(G2), np.max(B))
    print(f"Maximum pixel value across all channels: {max_val}")

    mu_matrix = estimate_base_signal(
        R, G1, G2, B, original_shape, max_val=max_val, channel_slices=channels
    )

    # numbers from paper
    # a_base = 1.0e-8
    # b_base = 1.0e-8

    # numbers from paper
    # a_target = 10.46e-5
    # b_target = 1.95e-6
    a_target = 16.966e-09
    b_target = 9.76e-09
    a_base, b_base = benchmark(cfa_linear, mu_matrix, a_target, b_target, image_name)
    inject_and_export_tiff(
        image_path, mu_matrix, a_base, b_base, a_target, b_target, image_name, seed=93
    )
    export_original_image_tiff(image_path, image_name)
    analyze_stego_difference(
        f"data/original_tiff/original_{image_name}.tiff",
        f"data/stego_output/stego_output_{image_name}.tiff",
        image_name,
    )


if __name__ == "__main__":
    main()
