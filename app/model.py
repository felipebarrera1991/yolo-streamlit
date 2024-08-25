from ultralytics import YOLO

def load_model(path, device):
    """
    Loads the YOLO model from the specified path and moves it to the given device.

    Parameters:
        path (str): The file path to the YOLO model (.pt file).
        device (str): The device to which the model should be moved ('cpu' or 'cuda').

    Returns:
        model_: The loaded YOLO model.
    """
    model_ = YOLO(path)
    model_.to(device)
    return model_

def infer_image(model, img, confidence):
    """
    Performs inference on the given image using the YOLO model.

    Parameters:
        model: The YOLO model used for inference.
        img (str or array): The path to the image file or the image array.
        confidence (float): The confidence threshold for object detection.

    Returns:
        image: The image with detected objects plotted on it.
    """
    model.conf = confidence
    result = model(img)
    image = result[0].plot()
    return image