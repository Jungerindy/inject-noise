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


def extract_raw_image(image_path):
    """
    Linearizes the raw image and de-interleaves the Bayer CFA.
    Returns the 4 color channels (R, G1, G2, B) and the original shape.
    """
    try:
        with rawpy.imread(image_path) as raw:
            cfa_array = raw.raw_image_visible.astype(np.float32)

            black_level = raw.black_level_per_channel[0]

            cfa_linear = np.maximum(cfa_array - black_level, 0)

            # Bayer Pattern (RGGB)
            R = cfa_linear[0::2, 0::2]
            G1 = cfa_linear[0::2, 1::2]
            G2 = cfa_linear[1::2, 0::2]
            B = cfa_linear[1::2, 1::2]

            print(f"Red Channel Shape: {R.shape}")
            return R, G1, G2, B, cfa_linear, cfa_linear.shape
    except Exception as e:
        print(f"Error processing raw image: {e}")
        return None, None, None, None, None


def estimate_base_signal(R, G1, G2, B, original_shape, max_val):
    """
    Applies BayesShrink wavelet denoising to each color channel independently
    to estimate the clean base signal (μ), then re-interleaves them.
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

    mu_matrix[0::2, 0::2] = R_mu
    mu_matrix[0::2, 1::2] = G1_mu
    mu_matrix[1::2, 0::2] = G2_mu
    mu_matrix[1::2, 1::2] = B_mu

    return mu_matrix


def benchmark(
    original_cfa,
    mu_matrix,
    a_base,
    b_base,
    a_target,
    b_target,
    picture_name,
    sensor_max=65535.0,
    seed=93,
):
    """
    Benchmarks the absolute measured variance of the original and stego images
    against the theoretical physical models.
    """
    np.random.seed(seed)

    mu_norm = mu_matrix / sensor_max
    var_delta_norm = (a_target * mu_norm + b_target) - (a_base * mu_norm + b_base)
    var_delta_norm = np.maximum(var_delta_norm, 0)

    injected_noise = (np.sqrt(var_delta_norm) * sensor_max) * np.random.randn(
        *mu_matrix.shape
    )

    original_noise = original_cfa - mu_matrix

    stego_noise_total = (original_cfa + injected_noise) - mu_matrix

    mu_flat = mu_matrix.flatten()
    orig_noise_flat = original_noise.flatten()
    stego_noise_flat = stego_noise_total.flatten()

    bins = np.linspace(0, np.max(mu_flat), 50)
    bin_indices = np.digitize(mu_flat, bins)

    meas_orig_var = []
    meas_stego_var = []
    bin_centers = []

    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) > 500:
            meas_orig_var.append(np.var(orig_noise_flat[mask]))
            meas_stego_var.append(np.var(stego_noise_flat[mask]))
            bin_centers.append((bins[i - 1] + bins[i]) / 2)

    base_a, base_b = estimate_sensor_parameters(bin_centers, meas_orig_var, sensor_max)
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
        f"Cover-Source Switching Absolute Benchmark\nBase({a_base}, {b_base}) -> Target({a_target}, {b_target})"
    )
    plt.xlabel("Pixel Brightness (\u03bc)")
    plt.ylabel("Absolute Noise Variance (\u03c3\u00b2)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(
        f"data/plots/benchmark_plot_{picture_name}.png", dpi=300, bbox_inches="tight"
    )


def inject_and_export_tiff(
    image_path,
    mu_matrix,
    a_base,
    b_base,
    a_target,
    b_target,
    output_filename="data/stego_output/stego_output.tiff",
    sensor_max=65535.0,
):
    """
    Injects the mathematical noise payload into the original raw sensor data
    using normalized physical parameters.
    """
    mu_norm = mu_matrix / sensor_max

    var_delta_norm = (a_target * mu_norm + b_target) - (a_base * mu_norm + b_base)
    var_delta_norm = np.maximum(var_delta_norm, 0)

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


def main():
    image_path = "data/Conversion_script/devHome/ALASKA_v2_RAWs/00002.dng"
    image_name = image_path.split("/")[-1].split(".")[0]
    metadata_path = "data/alaska_metadata.json"
    # metadata_df = read_metadata(metadata_path)
    iso_value = extract_base_iso(image_path)
    print(f"Extracted Base ISO: {iso_value}")

    R, G1, G2, B, cfa_linear, original_shape = extract_raw_image(image_path)
    max_val = max(np.max(R), np.max(G1), np.max(G2), np.max(B))
    print(f"Maximum pixel value across all channels: {max_val}")

    mu_matrix = estimate_base_signal(R, G1, G2, B, original_shape, max_val=max_val)

    # numbers from paper
    a_base = 1.0e-8
    b_base = 1.0e-8

    # numbers from paper
    a_target = 10.46e-5
    b_target = 1.95e-6

    benchmark(cfa_linear, mu_matrix, a_base, b_base, a_target, b_target, image_name)
    # inject_and_export_tiff(image_path, mu_matrix, a_base, b_base, a_target, b_target)


if __name__ == "__main__":
    main()
