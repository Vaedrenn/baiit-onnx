from __future__ import annotations

import os

import numpy as np
import onnxruntime as rt
from PIL import Image
from PyQt5.QtCore import QRunnable, QThreadPool


class Runnable(QRunnable):
    """
    Preprocesses images from directory using qt's multiprocessing model
    :param image_path: file name
    :param size: dimensions to resize to
    :param preprocessed_images: return array
    """

    def __init__(self, image_path, size, preprocessed_images):

        super().__init__()
        self.image_path = image_path
        self.size = size
        self.preprocessed_images = preprocessed_images

    def run(self):
        try:
            # Model only supports 3 channels
            image = Image.open(self.image_path).convert('RGB')

            # Pad image to square
            image_shape = image.size
            max_dim = max(image_shape)
            pad_left = (max_dim - image_shape[0]) // 2
            pad_top = (max_dim - image_shape[1]) // 2

            padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            padded_image.paste(image, (pad_left, pad_top))

            # Resize
            if max_dim != self.size:
                padded_image = padded_image.resize(
                    self.size,
                    Image.LANCZOS,
                )

            # Convert to numpy array
            image_array = np.asarray(padded_image, dtype=np.float32)

            # Convert PIL-native RGB to BGR
            image_array = image_array[:, :, ::-1]

            image_array = np.expand_dims(image_array, axis=0)

            # shape needs to be 4 dims

            self.preprocessed_images.append((self.image_path, image_array))

        except Exception as e:
            print(f"Runnable Error processing {self.image_path}: {e}")


def process_images_from_directory(model: rt.InferenceSession, directory: str) -> list[(str, np.ndarray)]:
    """
    Processes all images in a directory, does not go into subdirectories.
    Images need to be shaped before predict can be called on it.
    :param model: model, shape is used to resize of images
    :param directory: directory of images to be precessed
    :return: [(filename, ndarray)] returns a list of file names and processed images
    """
    preprocessed_images = []
    image_filenames = os.listdir(directory)
    pool = QThreadPool.globalInstance()

    # get dimensions from model
    _, height, width, _ = model.get_inputs()[0].shape
    size = (height, width)
    for filename in image_filenames:
        image_path = os.path.join(directory, filename)
        runnable = Runnable(image_path, size, preprocessed_images)
        pool.start(runnable)

    pool.waitForDone()
    return preprocessed_images
