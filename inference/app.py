from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def infer():
    data = request.json
    prompt = data.get("input", "")
    
    response_text = f"Echo: {prompt}"
    
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)