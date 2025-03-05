from flask import Flask, request, jsonify
import gradio as gr
from transformers import pipeline
import threading

# Initialize Flask app
app = Flask(__name__)

# Load sentiment analysis model (explicitly specifying the model)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Flask API Endpoint
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = sentiment_pipeline(text)
    return jsonify(result[0])  # Returns label & score

# Gradio Function
def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    return result[0]['label']

# Gradio Interface
gradio_interface = gr.Interface(
    fn=analyze_sentiment, 
    inputs="text", 
    outputs="text",
    title="Sentiment Analysis App",
    description="Enter a sentence and get the sentiment."
)

# Run Flask and Gradio Simultaneously
def run_flask():
    app.run(host="0.0.0.0", port=3000, debug=True, use_reloader=False)

if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
    gradio_interface.launch(share=True)
