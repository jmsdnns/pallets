import os
import PIL
import torch
from torchvision.transforms.functional import pil_to_tensor


# https://github.com/tnn1t1s/cpunks-10k
CPUNKS_DATA_DIR = "../../cpunks-10k/cpunks/data"
CPUNKS_IMAGE_DIR = "../../cpunks-10k/cpunks/images/training"

# total number of punks
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
    img_data = torch.concatenate(
        [i.reshape(i.shape[0], -1) for i in imgs],
        axis=-1
    )
    uniques = torch.unique(img_data, dim=1)

    return uniques.T


def get_punk_colors():
    punks = [get_punk_tensor(i) for i in range(CPUNKS_SIZE)]
    colors = many_image_colors(punks)
    return colors

