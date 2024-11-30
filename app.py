from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import base64
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

app = Flask(__name__)

CLASSES = [
    "OK",'Furcation'#, 'Grade_1', 'Grade_2', 'Grade_3'
] 

NUM_CLASSES = 2

detection_threshold = 0.8 

def create_model_FRCCN_RESNET50(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes = num_classes, pretrained=False, pretrained_backbone = True)
    return model

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = create_model_FRCCN_RESNET50(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load('./model703.pth', map_location=device))
model.eval()

def process_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(io.BytesIO(image_bytes))
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

@app.route('/')
def home():
    return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print(f'Files are',request.files)
        file = request.files['image']
        img_bytes = file.read()
        image_tensor = process_image(img_bytes)
        image_tensor = image_tensor
        with torch.no_grad():
            outputs = model(image_tensor)
        print(outputs)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        score_above_th = scores[scores >= detection_threshold].astype(float)
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        result = []
        for j, box in enumerate(boxes):
            result.append({
                'class': pred_classes[j],
                'score': float(score_above_th[j]),
                'bbox': box.tolist(),
            })
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
