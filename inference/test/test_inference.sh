curl -X POST http://10.74.0.21:5000/infer \
     -H "Content-Type: application/json" \
     -d '{"input": "Explain black holes", "length": 100}'
