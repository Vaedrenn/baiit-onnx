from __future__ import annotations

import os
from typing import Any

import numpy as np
import onnxruntime
import onnxruntime as rt
from PIL import Image

from src.commands.load_actions import load_labels, load_model
from src.commands.process_images import process_images_from_directory


def predict(
        model: rt.InferenceSession,
        labels: dict,
        thresholds: dict,
        image: np.ndarray
) -> dict[Any, list[list[Any] | Any]] | None:
    """
    Predicts tags for the image given the model and tags.
    :param model: model to use
    :param labels: {"category":[indexes], "category":[2,3,4,5]}
    :param image: processed image
    :param thresholds: {"category": threshold float, "general" : 0.5}
    :return: None if there are no tags within threshold otherwise returns:
    list
    """
    try:
        # Make a prediction using the model
        input_name = model.get_inputs()[0].name

        output_node = model.get_outputs()[0].name  # Onnx can output to different nodes, we only want the end output

        probs = model.run([output_node], {input_name: image})[0]

        # assign probs to tag names
        tag_names = list(zip(labels["tags"], probs[0].astype(float)))  # labels[tags] is the list of all tags

        ret_thing = {}

        for category, indexes in labels.items():
            # {category: [(tag, float)], 'rating':[('general', 0.43), ('sensitive', 0.63), ('questionable', 0.01)]
            # Get all names from indexes if it is in index
            if category != 'tags':
                tag_probs = [tag_names[i] for i in indexes if tag_names[i][1] > thresholds[category]]
                ret_thing[category] = tag_probs
        return ret_thing

    # unprocessed image
    except TypeError:
        print("Images need to be processed before calling this function, Call process_images_from_directory")
        return None

    except AttributeError:
        print("Channels must be 3, use  image = PIL.Image.open(img_path).convert('RGB')")
        return None


def predict_all(model: rt.InferenceSession,
                labels: dict,
                thresholds: dict,
                directory: str | os.path
                ) -> list[tuple[Any, dict[Any, list[list[Any] | Any]]]] | None:
    """
    Calls process_images_from_directory and predict on all images in the folder
    :param thresholds: kv pair of category names and thresholds{"rating": 0.0, "general": 0.5, "characters": 0.85}
    :param model: model to use
    :param labels: {"category":[indexes], "category":[2,3,4,5]}
    :param directory: folder to process
    :return: None if there are no tags within threshold otherwise returns:
    """

    images = process_images_from_directory(model, directory)

    processed_images = []
    for image in images:
        result = predict(model, labels, thresholds, image[1])
        if result is not None:
            processed_images.append((image[0], result))
    if processed_images is not None:
        return processed_images
    else:
        print("No results")
        return None


def predict_parallel(model, labels, thresholds, images):
    pass


def test_predict():
    path = r"C:\Users\khei\PycharmProjects\models\wd-vit-tagger-v3"
    model = load_model(path)
    test_dict = {"rating": 9, "general": 0, "characters": 4}
    thresh_dict = {"rating": 0.0, "general": 0.5, "characters": 0.85}
    labels = load_labels(path, "selected_tags.csv", test_dict)

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
    size = (height, width)  # Resize

    if max_dim != size:
        padded_image = padded_image.resize(
            size,
            Image.BICUBIC,
        )

    # Convert to numpy array
    image_array = np.asarray(padded_image, dtype=np.float32)

    # Convert PIL-native RGB to BGR
    image_array = image_array[:, :, ::-1]

    image_array = np.expand_dims(image_array, axis=0)

    ape = predict(model, labels, thresh_dict, image_array)

    for k, v in ape.items():
        print(k, v)


def test_predict_all():
    path = r"C:\Users\khei\PycharmProjects\models\wd-vit-tagger-v3"
    model = load_model(path)
    print(rt.get_device())
    test_dict = {"rating": 9, "general": 0, "characters": 4}
    thresh_dict = {"rating": 0.0, "general": 0.5, "characters": 0.85}
    labels = load_labels(path, "selected_tags.csv", test_dict)
    img_path = r"C:\Users\khei\PycharmProjects\baiit-onnx\tests\images"

    results = predict_all(model, labels, thresh_dict, img_path)
    for r in results:
        print(r)


if __name__ == '__main__':
    # test_predict()
    test_predict_all()
