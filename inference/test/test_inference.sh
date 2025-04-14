curl -X POST http://10.71.8.124:5000/infer \
     -H "Content-Type: application/json" \
     -d '{"input": "Explain black holes", "length": 100, "model": "llama_8B"}'
