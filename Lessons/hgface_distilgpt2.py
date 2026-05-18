from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

result = generator(
    "Python is",
    max_length=40
)

print(result)
