# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:53:00 2023

@author: Ashkan
"""

import torch
import cv2
import json
import os

# ## 1. Video chunking: 
#     chunk size : can be determined based on the available RAM and the expected frame rate of the video. 
#     For example, expected frame rate = 20 fps, video chunk size = 10 seconds could be used, then, 200 frames per chunk.

# ## 2. JSON response:
# Once all the chunks have been processed, the results can be combined into a single JSON response. The result of each chunck can be written into a temporary file. These temporary files can be read back in and combined into a single JSON response. 

# Set up the YoloV5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Set up the video reader
video_path = 'sample.mp4'

cap = cv2.VideoCapture(video_path)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up the chunk size and file paths
video_chunk_size = 4   # number of frames to process in each chunk
num_chunks = (num_frames // video_chunk_size) + 1  # total number of chunks
predictions_file = 'file.json'
if os.path.exists(predictions_file):
    os.remove(predictions_file)
    
for i in range (20): # range(num_chunks):
    # Read the frames for this chunk
    frames = []
    for j in range(video_chunk_size):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Process the frames using YoloV3
    predictions = model(frames)

    # Write the predictions to disk
    with open(predictions_file, 'a') as f:
        for k in range(len(predictions)):
            boxes = predictions.xyxy[k].tolist()
            frame_num = (i * video_chunk_size) + k
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                prediction = {'frame_id': frame_num, 'box': [x1, y1, x2, y2], 'confidence': conf, 'class': int(cls)}
                f.write(json.dumps(prediction) + '\n')
                
            # Draw the bounding boxes on the frame
            for box in boxes:
                x1, y1, x2, y2, _, cls = box
                color = (0, 0, 255) if cls == 0 else (0, 255, 0)  # red for person, green for other objects
                cv2.rectangle(frames[k], (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Show the frame with the bounding boxes
            cv2.imshow('Frame', frames[k])
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
                
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Read the predictions from disk and combine them into a single JSON object
results = []
with open(predictions_file, 'r') as f:
    for line in f:
        results.append(json.loads(line))

# Convert the results to a single JSON object
json_results = json.dumps(results)

# Return the JSON object to the client
print(json_results)
