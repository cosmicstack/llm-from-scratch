import urllib.request
import re

url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
file_path = "the-verdict.txt"

urllib.request.urlretrieve(url, file_path)

with open(file_path, "r", encoding="utf-8") as file:
    raw_text = file.read()

print("Total number of character: {}".format(len(raw_text)))
print(raw_text[:99])

def preprocess_text(text):
    result = re.split(r"([,.:;?_!\"()\']|--|\s+)", text)
    result = [item for item in result if item.strip()]
    return result

preprocessed_text = preprocess_text(raw_text)
print(preprocessed_text[:30])

def build_vocab(token_list):
    sorted_unique_tokens = sorted(set(token_list))
    vocab = {token: idx for idx, token in enumerate(sorted_unique_tokens)}
    return vocab

vocab = build_vocab(preprocessed_text)
for idx, item in enumerate(vocab.items()):
    print(f"{idx}: {item}")
    if idx == 50:
        break