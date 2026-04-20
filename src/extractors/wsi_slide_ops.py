import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import openslide
from PIL import Image, ImageFile
from scipy import ndimage
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def get_wsi_max_resolution(wsi_path):
    slide = openslide.OpenSlide(wsi_path)

    mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, "0.252"))
    mpp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, "0.252"))

    level_dimensions = slide.level_dimensions
    max_resolution = level_dimensions[0]
    return max_resolution, mpp_x, mpp_y


def resample_wsi(wsi_path, target_mpp, tile_size=4096, is_rgb=False, max_workers=None):
    slide = openslide.OpenSlide(wsi_path)
    max_resolution, mpp_x, mpp_y = get_wsi_max_resolution(wsi_path)

    scale_factor_x = mpp_x / target_mpp
    scale_factor_y = mpp_y / target_mpp

    new_width = int(max_resolution[0] * scale_factor_x)
    new_height = int(max_resolution[1] * scale_factor_y)

    downsample_needed = 1.0 / scale_factor_x
    best_level = slide.get_best_level_for_downsample(downsample_needed)

    level_downsample = slide.level_downsamples[best_level]
    relative_scale_x = scale_factor_x * level_downsample
    relative_scale_y = scale_factor_y * level_downsample

    mode = "RGB" if is_rgb else "L"
    resampled_image = Image.new(mode, (new_width, new_height))
    lock = threading.Lock()

    level_w, level_h = slide.level_dimensions[best_level]
    total_tiles = (level_w // tile_size + 1) * (level_h // tile_size + 1)

    def process_tile(x, y):
        try:
            w = min(tile_size, level_w - x)
            h = min(tile_size, level_h - y)

            x_lvl0 = int(x * level_downsample)
            y_lvl0 = int(y * level_downsample)

            tile = slide.read_region((x_lvl0, y_lvl0), best_level, (w, h))

            if not is_rgb:
                tile = tile.convert("L")
            else:
                tile = tile.convert("RGB")

            new_tile_w = int(w * relative_scale_x)
            new_tile_h = int(h * relative_scale_y)

            if new_tile_w > 0 and new_tile_h > 0:
                if new_tile_w != w or new_tile_h != h:
                    tile = tile.resize((new_tile_w, new_tile_h), Image.BICUBIC)

                if not is_rgb:
                    tile = tile.point(lambda p: 255 if p == 0 else p)

                paste_x = int(x * relative_scale_x)
                paste_y = int(y * relative_scale_y)

                with lock:
                    resampled_image.paste(tile, (paste_x, paste_y))
        except Exception:
            pass

        return 1

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_tile, x, y)
            for x in range(0, level_w, tile_size)
            for y in range(0, level_h, tile_size)
        ]

        for _ in tqdm(as_completed(futures), total=total_tiles, desc="Resampling WSI", leave=False):
            pass

    return resampled_image


def get_largest_tissue_region_center(wsi_path, target_mpp, crop_size=None):
    slide = openslide.OpenSlide(wsi_path)
    w, h = slide.dimensions

    lowest_level = slide.level_count - 1
    low_w, low_h = slide.level_dimensions[lowest_level]

    low_res_region_rgb = slide.read_region((0, 0), lowest_level, (low_w, low_h)).convert("RGB")
    low_res_np_gray = np.array(low_res_region_rgb.convert("L"))

    background_thresh = 210
    artifact_thresh = 10

    mask = (low_res_np_gray < background_thresh) & (low_res_np_gray > artifact_thresh)
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))

    labeled_array, num_features = ndimage.label(mask)
    if num_features == 0:
        print(f"Warning: No tissue detected in {os.path.basename(wsi_path)}. Using geometric center.")
        _, mpp_x, mpp_y = get_wsi_max_resolution(wsi_path)
        scale_factor_x = mpp_x / target_mpp
        scale_factor_y = mpp_y / target_mpp
        return int((w * scale_factor_x) // 2), int((h * scale_factor_y) // 2)

    sizes = ndimage.sum(mask, labeled_array, range(num_features + 1))
    largest_label = sizes.argmax()
    largest_region = labeled_array == largest_label

    com_y, com_x = ndimage.center_of_mass(largest_region)

    rel_x = com_x / low_w
    rel_y = com_y / low_h

    _, mpp_x, mpp_y = get_wsi_max_resolution(wsi_path)
    scale_factor_x = mpp_x / target_mpp
    scale_factor_y = mpp_y / target_mpp

    new_width = int(w * scale_factor_x)
    new_height = int(h * scale_factor_y)

    center_x_resampled = int(rel_x * new_width)
    center_y_resampled = int(rel_y * new_height)

    return center_x_resampled, center_y_resampled


def extract_crop_locally(wsi_path, target_mpp, crop_size, center_x, center_y, is_rgb=True):
    slide = openslide.OpenSlide(wsi_path)
    _, mpp_x, mpp_y = get_wsi_max_resolution(wsi_path)

    scale_x = mpp_x / target_mpp
    scale_y = mpp_y / target_mpp

    half_crop = crop_size // 2
    t_left = center_x - half_crop
    t_top = center_y - half_crop

    l0_left = int(t_left / scale_x)
    l0_top = int(t_top / scale_y)
    l0_box_w = int(crop_size / scale_x)
    l0_box_h = int(crop_size / scale_y)

    real_x = max(0, l0_left)
    real_y = max(0, l0_top)

    downsample_needed = 1.0 / scale_x
    best_level = slide.get_best_level_for_downsample(downsample_needed)
    level_downsample = slide.level_downsamples[best_level]

    read_w = int(l0_box_w / level_downsample) + 2
    read_h = int(l0_box_h / level_downsample) + 2

    try:
        region = slide.read_region((real_x, real_y), best_level, (read_w, read_h))
        region = region.convert("RGB" if is_rgb else "L")

        target_w_region = int(read_w * level_downsample * scale_x)
        target_h_region = int(read_h * level_downsample * scale_y)
        region = region.resize((target_w_region, target_h_region), Image.BICUBIC)

        final = Image.new("RGB" if is_rgb else "L", (crop_size, crop_size), (255, 255, 255) if is_rgb else 255)
        left = max(0, (region.size[0] - crop_size) // 2)
        top = max(0, (region.size[1] - crop_size) // 2)
        crop = region.crop((left, top, left + min(crop_size, region.size[0]), top + min(crop_size, region.size[1])))
        paste_x = max(0, (crop_size - crop.size[0]) // 2)
        paste_y = max(0, (crop_size - crop.size[1]) // 2)
        final.paste(crop, (paste_x, paste_y))
        return final
    except Exception as exc:
        raise RuntimeError(f"Failed to extract local crop from {wsi_path}: {exc}") from exc
