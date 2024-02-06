import io
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import mediapipe as mp
import imageio
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import uvicorn

app = FastAPI()

base_options = python.BaseOptions(model_asset_path='selfie_segmenter.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)



BG_COLOR = (0, 0, 0)  # black background
MASK_COLOR = (255, 255, 255)  # white mask

# mp_selfie_segmentation = mp.solutions.selfie_segmentation
# selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

@app.post("/remove_bg")
async def remove_bg(file: UploadFile = File(...)):
    imageMat = cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    imageNArray = cv2.cvtColor(imageMat, cv2.COLOR_BGR2RGBA)
    
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        image = mp.Image( image_format=mp.ImageFormat.SRGBA, data=imageNArray)
        # image = mp.Image.create_from_file(UPLOAD_FOLDER + file.filename)
        # image = mp.packet_creator.create_image_frame(image_format=mp.ImageFormat.SRGB, data=b)
        segmentation_result = segmenter.segment(image = image)
        category_mask = segmentation_result.category_mask
        # return {"shapes":category_mask.numpy_view().tolist()}
        # Generate a solid color image for the foreground (masked region)
        fg_image = np.zeros_like(image.numpy_view(), dtype=np.uint8)
        # print(category_mask.numpy_view())
        # return FileResponse(UPLOAD_FOLDER + file.filename)
        # return {"data":category_mask.numpy_view().tolist()}
        fg_image[category_mask.numpy_view() <= 0.2] = image.numpy_view()[category_mask.numpy_view() <= 0.2]
        
        # return StreamingResponse(io.BytesIO(fg_image.tobytes()), media_type="image/png")
        # Save the output image without the alpha channel
        image_bytes_io = io.BytesIO()
        imageio.imwrite(image_bytes_io, fg_image[:, :, :4],format="PNG")  # Remove the alpha channel
        return StreamingResponse(io.BytesIO(image_bytes_io.getbuffer()), media_type="image/png")
        
        # print(f'Background removed and saved as {output_filename}')
    # results = selfie_segmentation.process(image)
    # mask = results.segmentation_mask

    # bg_image = np.zeros(image.shape, dtype=np.uint8)
    # bg_image[:] = (0, 255, 0)  # Set background to green (replace with desired color/image)

    # output_image = cv2.bitwise_and(image, image, mask=mask)
    # output_image = cv2.addWeighted(output_image, 1, bg_image, 0.5, 0)

    # _, encoded_image = cv2.imencode('.png', output_image)
    # return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/png")


@app.get("/")
def read_root():
    return {"Hello": "World"}

UPLOAD_FOLDER = "uploads/"
@app.post("/removebg0")
async def removebg(file: UploadFile = File(...)):
    contents = await file.read()
    # Write the file contents to a new file on the server
    with open(UPLOAD_FOLDER + file.filename, "wb") as f:
        f.write(contents)
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        # Create the MediaPipe image file that will be segmented
        image = mp.Image.create_from_file(UPLOAD_FOLDER + file.filename)
        print(image)
        # Retrieve the masks for the segmented image
        segmentation_result = segmenter.segment(image)
        category_mask = segmentation_result.category_mask
        return {"shapes":category_mask.numpy_view()}
        # Generate a solid color image for the foreground (masked region)
        # fg_image = np.zeros_like(image.numpy_view(), dtype=np.uint8)
        # print(category_mask.numpy_view())
        # return FileResponse(UPLOAD_FOLDER + file.filename)
        # return {"data":category_mask.numpy_view().tolist()}
        # fg_image[category_mask.numpy_view() <= 0.2] = image.numpy_view()[category_mask.numpy_view() <= 0.2]
        
        # Save the output image without the alpha channel
        # imageio.imwrite(output_filename, fg_image[:, :, :3])  # Remove the alpha channel
        # print(f'Background removed and saved as {output_filename}')
    # return {"filename": file.filename}
@app.post("/upload")
async def upload_file(files: List[UploadFile] = File(...)):
    uploaded_files = []
    
    for file in files:
        contents = await file.read()
        file_path = UPLOAD_FOLDER + file.filename
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        uploaded_files.append({"filename": file.filename, "file_path": file_path})
    
    return uploaded_files
if __name__=="__main__":
    config = uvicorn.Config("main:app", port=5000, log_level="info")
    server = uvicorn.Server(config)
    server.run()  