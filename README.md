# FastAPI Background Removal App

This FastAPI application removes the background from an image using MediaPipe, NumPy, and OpenCV (cv2).

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/biswasray/pyar.git
    cd your_project
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start the FastAPI app using:

    ```bash
    python main.py
    ```

2. Once the server is running, you can access the FastAPI documentation at [http://localhost:8000/docs](http://localhost:8000/docs) to explore the available endpoints.

3. To remove the background of an image, send a POST request to `/remove_bg` with the image file as a form-data parameter named `file`.

## Example

Here's an example using cURL to remove the background of an image:

```bash
curl -X POST -F "file=@/path/to/your/image.jpg" http://localhost:8000/remove_bg
```

## Reference
- [MediaPipe docs](https://developers.google.com/mediapipe/api/solutions/python/mp)