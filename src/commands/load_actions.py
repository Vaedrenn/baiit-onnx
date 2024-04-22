from __future__ import annotations

import os

import onnxruntime as rt


def load_model(model_path: str | os.path) -> rt.InferenceSession:
    """
    Loads models model_path, should be called before using predict
    :param model_path: file name
    :return: returns the loaded model
    """
    file_name = "model.onnx"
    path = os.path.join(model_path, file_name)

    try:
        print("Loading model")
        model = rt.InferenceSession(path)
    except FileNotFoundError as file_not_found_error:
        print("Model file not found:", file_not_found_error)
        return None
    except (IOError, OSError) as io_os_error:
        print("Error loading model:", io_os_error)
        return None
    print("Finished Loading models")
    return model

