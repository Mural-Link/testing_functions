import azure.functions as func # type: ignore
import logging
import numpy as np
import cv2
from PIL import Image
import io
import base64
import requests
import json
import random

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def expand_mask(mask, expand_pixels=2):
    kernel = np.ones((expand_pixels*2+1, expand_pixels*2+1), np.uint8)
    expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return expanded_mask

def process_image_with_target_color(target_hex_color, wall_mask, image_rgb):
    target_color = np.array(hex_to_rgb(target_hex_color), dtype=np.uint8)
    image_hsv = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2HSV)
    _, _, v_channel = cv2.split(image_hsv)
    v_rgb = cv2.cvtColor(v_channel, cv2.COLOR_GRAY2RGB)
    target_color_image = np.full(v_rgb.shape, target_color, dtype=np.uint8)
    normalized_v_channel = v_channel / 255.0
    enhancement = np.where(
        normalized_v_channel > 0.9, 0.0,
        np.where(
            normalized_v_channel < 0.3, 0.4,
            0.00 + (0.4 - 0.0) * (0.9 - normalized_v_channel) / (0.9 - 0.3)
        )
    )
    enhancement = enhancement[:, :, np.newaxis]
    blended_image = cv2.multiply(v_rgb, target_color_image, scale=1/255.0) * (1 + enhancement)
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

    expanded_mask = expand_mask(wall_mask)

    final_image = np.where(expanded_mask[:, :, np.newaxis], blended_image, np.array(image_rgb))
    return final_image

def apply_colors(image, masks, colors):
    image_rgb = np.array(image.convert('RGB'))
    processed_image = image_rgb.copy()

    for mask, color in zip(masks, colors):
        if color:
            mask_np = mask
            processed_image = process_image_with_target_color(color, mask_np, processed_image)

    return Image.fromarray(processed_image)


@app.route(route="http_trigger_1")
def http_trigger_1(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    try:
        # Get the image data from the request
        image_data = req.get_body()
        
        # Call the Image Processing Function
        processing_function_url = "https://image-segment-app.azurewebsites.net/api/image_segment?code=i8v1QIf4WAXuIOlIN7DAObZEjHBLdAsFZOUNY4Kmz6IqAzFu2mnfow%3D%3D"
        response = requests.post(processing_function_url, data=image_data, 
                                 headers={'Content-Type': 'application/octet-stream'})
        
        if response.status_code != 200:
            return func.HttpResponse(f"Error calling Image Processing Function: {response.text}", status_code=500)
        
        # Parse the response
        result = response.json()
        image = np.array(Image.open(io.BytesIO(base64.b64decode(result['image']))))
        mask_segments = [np.frombuffer(base64.b64decode(mask), dtype=np.uint8).reshape(image.shape[:2]) 
                         for mask in result['mask_segments']]
        
        # Generate random colors for each segment
        color_choices = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) 
                         for i in range(len(mask_segments))]

        # Apply colors
        colored_image = apply_colors(Image.fromarray(image), mask_segments, color_choices)

        # Convert the colored image to base64
        buffered = io.BytesIO()
        colored_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Return the colored image
        return func.HttpResponse(
            json.dumps({'colored_image': img_str}),
            mimetype="application/json",
            status_code=200
        )
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return func.HttpResponse(
            f"An error occurred: {str(e)}",
            status_code=500
        )