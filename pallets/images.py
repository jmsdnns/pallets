import os

import PIL
import torch
from torchvision.transforms.functional import pil_to_tensor

from . import paths

CPUNKS_IMAGE_DIR = os.path.join(paths.CPUNKS_ROOT, "data", "images")
CPUNKS_SIZE = 10000


def get_punk_path(id):
    return f"{CPUNKS_IMAGE_DIR}/punk{id:#04d}.png"


def get_punk(id):
    """
    Loads an image from the punks dataset as an RGB PIL image.
    """
    image_path = get_punk_path(id)
    if not os.path.exists(image_path):
        raise Exception(f"ERROR: image doesn't exist {image_path}")
    pil_img = PIL.Image.open(image_path)
    return pil_img


def get_punk_tensor(id):
    """
    Same thing as `get_punk`, but returns a tensor
    """
    image = get_punk(id)
    return pil_to_tensor(image) / 255


def one_image_colors(img):
    """
    Returns an array of unique colors found in an image array
    """
    img = img.reshape(img.shape[0], -1)
    uniques = torch.unique(img, dim=1)

    return uniques.T


def many_image_colors(imgs):
    """
    Returns an array of unique colors found in an array of images arrays
    """
    img_data = torch.concatenate([i.reshape(i.shape[0], -1) for i in imgs], axis=-1)
    uniques = torch.unique(img_data, dim=1)

    return uniques.T


def get_punk_colors():
    punks = [get_punk_tensor(i) for i in range(CPUNKS_SIZE)]
    colors = many_image_colors(punks)
    return colors


def has_colors(img_colors, colors):
    """
    Checks all colors in img_colors for a match with `colors`
    """
    matches = []
    for ic in img_colors:
        for c in colors:
            if (ic == c).tolist() == [True, True, True, True]:
                matches.append(ic)
    return matches


def find_pixels(img, colors):
    """
    Finds every pixel in `img` with color values that matches `colors`
    """
    img = img.reshape(img.shape[0], -1)
    y_pixels = []
    for pixel_idx in range(img.shape[1]):
        pixel = torch.tensor(
            [
                img[0][pixel_idx],
                img[1][pixel_idx],
                img[2][pixel_idx],
                img[3][pixel_idx],
            ]
        )
        coords = (pixel_idx // 24), (pixel_idx % 24)
        matches = has_colors(pixel.unsqueeze(0), colors)
        if len(matches) > 0:
            y_pixels.append((coords, matches[0]))
    return y_pixels


def map_colors(imgs, colors):
    """
    Finds every pixel in `imgs` with a color value that matches `colors` and
    keeps a list of every instance found, including duplicates. A tally of
    occurrences per pixel is also created for use in heatmaps.
    """
    occurrences = dict()
    presence = dict()
    for img in imgs:
        matches = find_pixels(img, colors)
        for m in matches:
            coords, yellow = m
            occurrences[coords] = occurrences.get(coords, 0) + 1
            presence[coords] = presence.get(coords, []) + [yellow]
    return occurrences, presence
