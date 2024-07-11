import cv2
import numpy as np
import onnxruntime as ort

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle




# this function converts the image into onnx format
def preprocess(image, input_shape):
    h, w = input_shape
    resized = cv2.resize(image, (w, h))

    # Normalization (between [0, 1])
    img = resized.astype(np.float32) / 255.0

    # Convert HWC to CHW format
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def capture_image():

    # may need to change this later for raspberry pi, 
    # currently 0 is the default camera
    cap = cv2.VideoCapture(0)

    # test if the camera is opened
    if not cap.isOpened():
        raise Exception("Could not open video device")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise Exception("Failed to capture image")

    height, width, _ = frame.shape

    # Currently do center crop (640x640)
    
    # please be aware that rover code used previously did not do center crop
    center_x, center_y = width // 2, height // 2
    crop_size = 640 # input in the onnx that we use

    start_x = max(center_x - crop_size // 2, 0)
    start_y = max(center_y - crop_size // 2, 0)
    end_x = min(center_x + crop_size // 2, width)
    end_y = min(center_y + crop_size // 2, height)

    # Ensure the crop size does not exceed the frame dimensions
    cropped_frame = frame[start_y:end_y, start_x:end_x]

    return cropped_frame



if __name__ == "__main__":


    image = capture_image()


    # change this to your model path
    model_path = 'best_nano10.onnx'
    session = ort.InferenceSession(model_path)


    # change this to your configured input shape
    input_shape = (640, 640)  
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Image Preporcessing
    height, width = image.shape[:2]
    preprocessed_image = preprocess(image, input_shape)


    # Model Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: preprocessed_image})


    # Model Results Extraction
    scores = [outputs[0][ 0, :, -2]]
    score_array = scores[0]

    # Change the score threshold here
    idxs = [index for index, score in enumerate(score_array) if score> 0.001]

    reshaped_outputs = outputs[0][0,:,:][idxs]
    bboxs = reshaped_outputs[:,:4]
    confidences = reshaped_outputs[:,4]
    ids = reshaped_outputs[:,5]


    # Plot the image
    for box, score, class_id in zip(bboxs, confidences, ids):
        model_input = 640
        x1, y1, x2, y2 = map(lambda v: int(v) / model_input, box) # convert to absolute positions

        height, width = image.shape[:2] #160, 160 # image.shape[:2]
        x1_scaled = x1 * width
        y1_scaled = y1 * height
        x2_scaled = x2 * width
        y2_scaled = y2 * height

        box_width = x2_scaled - x1_scaled
        box_height = y2_scaled - y1_scaled

        plt.gca().add_patch(Rectangle((x1_scaled, y1_scaled), box_width, box_height, edgecolor='red',
                        facecolor='none', lw=2))


        # Add label
        label = f'Class {int(class_id)}: {score:.2f}'
        plt.text(x1, y1 - 10, label, color='g', fontsize=8, ha='left', va='center')

plt.show()