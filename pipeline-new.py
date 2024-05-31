import json
import requests
import sys
import os
from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, LayoutLMModel, LayoutLMConfig
from datetime import date
import torch
from bs4 import BeautifulSoup

# Load models
t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
layoutlm = LayoutLMModel(LayoutLMConfig())

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

def replace_images(detected_images, user_input):
    pexels_api_key = "YOUR_PEXELS_API_KEY"
    replaced_images = {}
    for image_name, _ in detected_images.items():
        image_url = "img/search?query={user_input['Industry']}"
        headers = {"Authorization": f"Bearer {pexels_api_key}"}
        response = requests.get(image_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("photos"):
                replaced_images[image_name] = data["photos"][0]["src"]["medium"]
    return replaced_images

def replace_content(template_content, user_input):
    input_text = f"replace: {template_content} with: {json.dumps(user_input)}"
    input_ids = t5.tokenizer.encode(input_text, return_tensors="pt")
    outputs = t5.generate(input_ids)
    replaced_content = t5.tokenizer.decode(outputs[0], skip_special_tokens=True)

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

def main(user_input):
    template_path = choose_template(user_input)

    with open(template_path, "r") as f:
        template_content = f.read()

    detected_images = detect_images(template_content)
    replaced_images = replace_images(detected_images, user_input)
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

