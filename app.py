from flask import Flask, request, jsonify
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F

app = Flask(__name__)

# load model once
model = InceptionResnetV1(pretrained='vggface2').eval()

# load your saved embeddings
database = torch.load('face_database.pt', map_location='cpu')

# transform
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.route('/')
def home():
    return "Face Recognition API is running!"

@app.route('/identify', methods=['POST'])
def identify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')

    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        new_embedding = model(tensor)

    # find best match
    best_match = None
    best_score = -1.0
    for name, emb in database.items():
        sim = F.cosine_similarity(new_embedding, emb).item()
        if sim > best_score:
            best_score = sim
            best_match = name

    threshold = 0.7
    if best_match is None or best_score < threshold:
        return jsonify({'result': 'Not Matched'})

    return jsonify({'result': best_match, 'score': float(best_score)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
