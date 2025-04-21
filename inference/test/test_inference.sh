curl -X POST http://10.72.10.19:5000/infer \
     -H "Content-Type: application/json" \
     -d '{"input": "Explain black holes", "length": 100, "model": "llama_8B"}'
