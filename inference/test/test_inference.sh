curl -X POST http://10.71.8.128:5000/infer \
     -H "Content-Type: application/json" \
     -d '{"input": " is not defined NameError: name hello is not defined", "length": 256, "model": "llama_8B"}'
