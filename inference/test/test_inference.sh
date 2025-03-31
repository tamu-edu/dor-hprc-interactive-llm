curl -X POST http://10.72.10.12:5000/infer \
     -H "Content-Type: application/json" \
     -d '{"input": "Explain black holes", "length": 100}'
