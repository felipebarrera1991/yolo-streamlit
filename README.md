# Yolo Streamlit

## Overview

The **yolo-streamlit** is a web-based application that leverages YOLOv8 and YOLOv10 models for object detection in images and videos. Built using Streamlit, this dashboard allows users to easily upload and switch between pre-trained YOLO models, configure device options (CPU or CUDA), and perform object detection on their input data.

## Features

- **Model Selection**: Choose between YOLOv8 and YOLOv10 models for object detection.
- **Device Configuration**: Select the device (CPU or CUDA) to run the model, depending on the available hardware.
- **Input Types**: Process both images and videos for object detection.
- **Confidence Threshold**: Adjust the confidence threshold to control the sensitivity of the model's predictions.
- **Real-time Feedback**: View the processed image or video with detected objects highlighted in real-time.

## Installation

### Prerequisites

- Python 3.8 or higher
- [Streamlit](https://streamlit.io/) for the web interface
- [PyTorch](https://pytorch.org/) for loading and running YOLO models
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5) for the object detection models

### Setting Up the Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/felipebarrera1991/yolo-streamlit.git
   cd yolo-streamlit
   ```

2. **Set Up the Conda Environment**:
   ```bash
   conda env create -f environment.yaml
   conda activate yolo-streamlit-env
   ```

3. **Place YOLO models**:
   Ensure the YOLOv8 and YOLOv10 model files are placed in the `models/` directory:
   ```
   models/
       ├── yolov8n.pt
       └── yolov10n.pt
   ```

## Usage

1. **Run the application**:
   ```bash
   streamlit run app/main.py
   ```

2. **Access the dashboard**:
   Open your browser and go to `http://localhost:8501` to use the dashboard.

3. **Select model and device**:
   - Use the sidebar to choose between YOLOv8 and YOLOv10 models.
   - If CUDA is available, you can choose to run the model on the GPU for faster processing.

4. **Upload and process input**:
   - Upload an image or video file using the provided options.
   - Adjust the confidence threshold as needed.
   - The processed image or video with detected objects will be displayed on the right.

## Project Structure

```
yolo-streamlit/
│
├── app/
│   └── main.py                # Main Streamlit application script
│   └── model.py               # YOLO model loading and inference logic
│   └── input.py               # Input processing module handles image and video input
│
├── models/
│   ├── yolov8n.pt             # YOLOv8 model file
│   └── yolov10n.pt            # YOLOv10 model file
│
├── data/
│   ├── image/         # Directory for uploaded images
│   └── video/                 # Directory for uploaded videos
│
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## Contributing

Contributions are welcome! Please fork this repository, create a new branch, and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
