from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any
import onnxruntime as rt
import numpy as np
from PIL import Image
from PyQt5.QtCore import QRunnable, QThreadPool
from src.commands.load_actions import load_labels, load_model


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
                    (self.size, self.size),
                    Image.LANCZOS,
                )

            # Convert to numpy array
            image_array = np.asarray(padded_image, dtype=np.float32)

            # Convert PIL-native RGB to BGR
            image_array = image_array[:, :, ::-1]

            image_array = np.expand_dims(image_array, axis=0)

            self.preprocessed_images.append((self.image_path, image_array))

        except Exception as e:
            print(f"Error processing {self.image_path}: {e}")


def process_images_from_directory(model: rt.InferenceSession, directory: str | os.path) -> list[(str, np.ndarray)]:
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


def predict(
        model: rt.InferenceSession,
        labels: dict,
        image: np.ndarray,
        score_threshold: float = 0.5,
        char_threshold: float = 0.85
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float], str] | None:
    """
    Predicts tags for the image given the model and tags.
    :param model: model to use
    :param labels: {"category":[2,3,4,5], "category":[2,3,4,5]}
    :param image: processed image
    :param score_threshold: general tags, if the probability of the prediction is greater than this number add to tags
    :param char_threshold: character tags, see above
    :return: None if there are no tags within threshold otherwise returns:
    list
    """
    try:
        # Make a prediction using the model
        input_name = model.get_inputs()[0].name
        print("Input Node Name:", input_name)

        label_name = model.get_outputs()[0].name
        print("Output Node Name:", label_name)

        print("Shape of Input Data (image):", image.shape)  # Print the shape of the input data

        probs = model.run([label_name], {input_name: image})[0]

        labels = list(zip(labels[0], probs[0].astype(float)))  # labels[0] is the list of all tags


    # unprocessed image
    except TypeError:
        print("Images need to be processed before calling this function, Call process_images_from_directory")
        return None

    except AttributeError:
        print("Channels must be 3, use  image = PIL.Image.open(img_path).convert('RGB')")
        return None



if __name__ == '__main__':
    path = r"C:\Users\khei\PycharmProjects\models\wd-vit-tagger-v3"
    model = load_model(path)
    test_dict = {"rating": 9, "general": 0, "characters": 4}
    t = load_labels(path, "selected_tags.csv", test_dict)

    image_path = r'C:\Users\khei\PycharmProjects\baiit-onnx\tests\images\1670120513144187.png'
    # Model only supports 3 channels
    image = Image.open(image_path).convert('RGB')

    # Pad image to square
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    _, height, width, _ = model.get_inputs()[0].shape
    size = (height, width)    # Resize

    if max_dim != size:
        padded_image = padded_image.resize(
            size,
            Image.LANCZOS,
        )

    # Convert to numpy array
    image_array = np.asarray(padded_image, dtype=np.float32)

    # Convert PIL-native RGB to BGR
    image_array = image_array[:, :, ::-1]

    image_array = np.expand_dims(image_array, axis=0)

    predict(model, test_dict, image_array)