from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Initialize the Flask app
app = Flask(__name__)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-2-zephyr-1_6b")
model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-2-zephyr-1_6b")

# Create the pipeline for text generation
text_generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    if 'content' not in data:
        return jsonify({"error": "No content provided"}), 400
    
    try:
        # Generate text using the model
        generated_text = text_generation_pipeline(data['content'])[0]["generated_text"]
        return jsonify({"generated_text": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Welcome to the text generation API"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
