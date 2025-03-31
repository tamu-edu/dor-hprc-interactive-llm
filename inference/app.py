from flask import Flask, request, jsonify
from inference_script import perform_inference
app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def infer():
    try:
        data = request.json
        prompt = data.get("input", "")
        max_response_length = int(data.get("length", ""))
        return jsonify({"response": perform_inference(prompt, max_response_length)})
    except Exception as e:
        print("failed with exception: ", e)
        return jsonify({"status": 500})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
