import json
import requests
import torch
import sys
import os
from yolo import YOLO
import requests
from transformers import T5ForConditionalGeneration
t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
from datetime import date
from transformers import LayoutLMModel, LayoutLMConfig
configuration = LayoutLMConfig()
layoutlm = LayoutLMModel(configuration)
configuration = layoutlm.config
sys.path.append('/Users/stevepalmer/.local/pipx/venvs/flask/lib/python3.12/site-packages')

# Path to the site-packages directory of the Homebrew-managed Python
homebrew_site_packages = '/Users/stevepalmer/Library/Python/3.12/lib/python/site-packages'

# Add the Homebrew Python's site-packages directory to sys.path
sys.path.append(homebrew_site_packages)

from flask import Flask, request, jsonify

# Import statements for new functions
from image_processing import detect_image, replace_image

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/process', methods=['POST'])
def process_json():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Process the JSON data
        result = {'processed_data': data}

        # Return the result as JSON response
        return jsonify(result), 200

    except json.JSONDecodeError as e:
        return jsonify({'error': f'Error parsing JSON: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
# Step 1: LayoutLM chooses website template file based on user data input
def analyze_layoutlm():
    """
    Simulate LayoutLM analysis by selecting a template based on user input data.
    """
    input_data = request.args.get('input_data')
    print("Analyzing user input with LayoutLM...")

    # In a real scenario, integrate with LayoutLM model here
    # For now, we simulate by returning a predefined template

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
        <img src='IMAGE_URL_PLACEHOLDER' alt='Industry Image'>
        <p>Location: {Location}</p>
        <p>Industry: {Industry}</p>
        <p>Description: {Description}</p>
    </body>
    </html>
    """

    return selected_template.format(**input_data)
# choose template
def choose_template(user_input):
    industry = user_input.get('Industry', '').lower()
    if industry == 'technology':
        return 'templates/technology.html'
    elif industry == 'health':
        return 'templates/health.html'
    else:
        return 'templates/default.html'

# Step 2: YOLO detects images inside the selected website template
def detect_images(template_path):
    yolo = YOLO()  # Instantiate YOLO object
    detected_images = yolo.detect(template_path)  # Perform image detection
    return detected_images

# Step 3: Pexels API changes/replaces the detected images based on user input data {Industry}
def replace_images(detected_images, user_input):
    pexels_api_key = "YOUR_PEXELS_API_KEY"
    # Placeholder code to replace images using Pexels API
    replaced_images = {}
    for image_name in detected_images:
        image_url = f"https://api.pexels.com/v1/search?query={user_input['Industry']}"
        headers = {"Authorization": f"Bearer {pexels_api_key}"}
        response = requests.get(image_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("photos"):
                replaced_images[image_name] = data["photos"][0]["src"]["medium"]
    return replaced_images

# Step 4: T5 changes/replaces the written content of the website
def replace_content(template_path, user_input):
    t5_model = t5()  # Instantiate T5 object
    with open(template_path, "r") as f:
        template_content = f.read()
    # Placeholder code to replace content using T5
    replaced_content = t5_model.replace(template_content, user_input)
    # Add footer with copyright notice
    current_date = date.today().strftime("%Y-%m-%d")
    business_name = user_input.get("BusinessName", "Business Name")
    footer = f"<footer style='text-align: center;'>Copyright {current_date} {business_name} Made with AI by <a href='https://c4rrot.com'>Carrot</a></footer>"
    replaced_content += footer
    return replaced_content

# Step 5: Send final generated template to a subdomain
def send_to_subdomain(generated_template, business_name):
    # Placeholder code to send the template to a subdomain
    subdomain_url = f"www.{business_name}.carrotfullstack.com"
    with open(f"{subdomain_url}/index.html", "w") as f:
        f.write(generated_template)

# Main function to orchestrate the workflow
def main(user_input):
    # Step 1: Choose website template
    template_path = choose_template(user_input)

    # Step 2: Detect images in the template
    detected_images = detect_images(template_path)

    # Step 3: Replace images based on user input
    replaced_images = replace_images(detected_images, user_input)

    # Step 4: Replace content and add footer
    generated_template = replace_content(template_path, user_input)

    # Step 5: Send final generated template to subdomain
    business_name = user_input.get("BusinessName", "business")
    send_to_subdomain(generated_template, business_name)



@app.route('/run', methods=['POST'])

def run():

    # Extract user input from request

    request_json = request.get_json()

    if request_json and 'user_input' in request_json:

        user_input = request_json['user_input']



        # Perform main processing

        result = main(user_input)



        # Return result as JSON response

        return jsonify({'result': result})



    else:

        return jsonify({'error': 'Invalid request'})


# Entry point for the script
if __name__ == "__main__":
    # Example user input data
    user_input = {
        "BusinessName": "example",
        "Location": "City",
        "Industry": "technology",
        "Description": "Lorem ipsum dolor sit amet."
    }
    main(user_input)