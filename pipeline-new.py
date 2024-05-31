import json
import requests
import sys
import os
from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, LayoutLMModel, LayoutLMConfig, T5Tokenizer
from datetime import date
import torch
from torch.utils.data import DataLoader, Dataset
import torch
from bs4 import BeautifulSoup

# Load models
t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
layoutlm = LayoutLMModel(LayoutLMConfig())

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['input'], self.data[idx]['target']
    
def train_t5(train_data, num_epochs=3, batch_size=4, learning_rate=1e-4):
       
    # Prepare dataset and data loader
    dataset = CustomDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(t5.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        t5.train()
        total_loss = 0
        
        for input_seqs, target_seqs in dataloader:
            optimizer.zero_grad()
            
            input_ids = t5_tokenizer.batch_encode_plus(input_seqs, return_tensors="pt", padding=True, truncation=True)['input_ids']
            target_ids = t5_tokenizer.batch_encode_plus(target_seqs, return_tensors="pt", padding=True, truncation=True)['input_ids']
            
            outputs = t5(input_ids=input_ids, labels=target_ids)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
    
    # Save the trained model
    t5.save_pretrained("saved_models/t5_custom")
    t5_tokenizer.save_pretrained("saved_models/t5_custom")
    
    print("Training complete. Model saved.")
selected_template = """
    <html>
    <head><title>{Name}'s Website</title></head>
    <body>
        <h1>CONTENT_PLACEHOLDER</h1>
        <img src='img/default.jpeg' alt='Industry Image'>
        <p>Location: {Location}</p>
        <p>Industry: {Industry}</p>
        <p>Description: {Description}</p>
    </body>
    </html>
    """

# YOLO class definition
class YOLO:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def detect(self, image_path):
        results = self.model(image_path)
        return results.pandas().xyxy[0].to_dict(orient="records")

# Flask application setup
app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/process', methods=['POST'])
def process_json():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400
        result = {'processed_data': data}
        return jsonify(result), 200
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Error parsing JSON: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def analyze_layoutlm():
    input_data = request.args.get('input_data')
    print("Analyzing user input with LayoutLM...")

    # Simulating LayoutLM analysis for now
    input_data = {
        'Name': 'John Doe',
        'Location': 'City',
        'Industry': 'Technology',
        'Description': 'Lorem ipsum dolor sit amet.'
    }

    selected_template = """
    <html>
    <head><title>{Name}'s Website</title></head>
    <body>
        <h1>CONTENT_PLACEHOLDER</h1>
        <img src='img/default.jpeg' alt='Industry Image'>
        <p>Location: {Location}</p>
        <p>Industry: {Industry}</p>
        <p>Description: {Description}</p>
    </body>
    </html>
    """
    return selected_template.format(**input_data)

def detect_images(template_content):
    yolo = YOLO()  # Instantiate YOLO object

    # Parse HTML to extract image paths
    soup = BeautifulSoup(template_content, 'html.parser')
    image_tags = soup.find_all('img')
    image_paths = [img['src'] for img in image_tags if 'src' in img.attrs]

    detected_images = {}
    for image_path in image_paths:
        detected_images[image_path] = yolo.detect(image_path)
    
    return detected_images

def replace_images(template_content, detected_images, user_input):
    pexels_api_key = "https://www.pexels.com/@alexander-zeisl-1334539997/"
    replaced_images = {}

    for image_path in detected_images.keys():
        query = user_input['Industry']
        image_url = f"https://api.pexels.com/v1/search?query={query}"
        headers = {"Authorization": f"Bearer {pexels_api_key}"}
        response = requests.get(image_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("photos"):
                new_image_url = data["photos"][0]["src"]["medium"]
                replaced_images[image_path] = new_image_url
                template_content = template_content.replace(image_path, new_image_url)
    
    return template_content, replaced_images

def replace_content(template_content, user_input):
    input_text = f"replace: {template_content} with: {json.dumps(user_input)}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt")
    outputs = t5.generate(input_ids)
    print(input_text)
    replaced_content = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    current_date = date.today().strftime("%Y-%m-%d")
    business_name = user_input.get("BusinessName", "Business Name")
    footer = f"<footer style='text-align: center;'>Copyright {current_date} {business_name} Made with AI by <a href='https://c4rrot.com'>Carrot</a></footer>"
    replaced_content += footer
    return replaced_content

def send_to_subdomain(generated_template, business_name):
    subdomain_url = f"www.{business_name}.carrotfullstack.com"
    os.makedirs(subdomain_url, exist_ok=True)
    with open(f"{subdomain_url}/index.html", "w") as f:
        f.write(generated_template)

def choose_template(user_input):
    industry = user_input.get('Industry', '').lower()
    if industry == 'technology':
        return 'templates/technology.html'
    elif industry == 'health':
        return 'templates/health.html'
    else:
        return 'templates/default.html'

def main(user_input, train=False, train_data=None):
    train_data=f"replace: {selected_template} with: {json.dumps(user_input)}"
    if train:
        train_t5(train_data)
    template_path = choose_template(user_input)

    with open(template_path, "r") as f:
        template_content = f.read()
   
    detected_images = detect_images(template_content)
    template_content, replaced_images = replace_images(template_content, detected_images, user_input)
    generated_template = replace_content(template_content, user_input)    
    business_name = user_input.get("BusinessName", "business")
    send_to_subdomain(generated_template, business_name)
    return generated_template

@app.route('/run', methods=['POST'])
def run():
    request_json = request.get_json()
    if request_json and 'user_input' in request_json:
        user_input = request_json['user_input']
        result = main(user_input)
        return jsonify({'result': result})
    else:
        return jsonify({'error': 'Invalid request'})

if __name__ == "__main__":
    user_input = {
        "BusinessName": "example",
        "Location": "City",
        "Industry": "technology",
        "Description": "Lorem ipsum dolor sit amet."
    }
    main(user_input)

