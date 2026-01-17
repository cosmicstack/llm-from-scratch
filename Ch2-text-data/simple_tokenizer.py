import re
from typing import Dict, List
from xxlimited import new

class SimpleTokenizerV1:
    def __init__(self, vocab: Dict[str, int]):
        # vocab is prebuilt here
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        result = re.split(r"([,.:;?_!\"()\']|--|\s+)", text)
        result = [item for item in result if item.strip()]
        result = [item if item in self.str_to_int else "<|UNK|>" for item in result]
        ids = [self.str_to_int[s] for s in result]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[id] for id in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

# Writing this temporary class for a vocab builder
class VocabBuilder:
    # Obviously this is a super simple Class. I'm just passing one simple text file.
    # But, at prod level, how is the vocabulary built? 
    def __init__(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as file:
            self.text = file.read()

    def preprocess_text(self, text):
        result = re.split(r"([,.:;?_!\"()\']|--|\s+)", text)
        result = [item for item in result if item.strip()]
        return result
    
    def build_vocab(self):
        token_list = self.preprocess_text(self.text)
        sorted_unique_tokens = sorted(set(token_list))
        sorted_unique_tokens.extend(["<|EOS|>", "<|UNK|>"])
        vocab = {token: idx for idx, token in enumerate(sorted_unique_tokens)}
        return vocab

if __name__ == "__main__":
    vocab_builder = VocabBuilder("the-verdict.txt")
    vocab = vocab_builder.build_vocab()

    tokenizer = SimpleTokenizerV1(vocab)

    new_text = "She had tears in her eyes; I had my hands in my pockets without a clue."
    new_text_token_ids = tokenizer.encode(new_text)
    print(new_text_token_ids)

    print("Back to text:\n")
    print(tokenizer.decode(new_text_token_ids))

    print("=========== V2 ============\n\n")

    new_text1 = "Hello world wtf is happa?"
    new_text2 = "IDK brooo"
    new_text_joined = " <|EOS|> ".join((new_text1, new_text2))
    print(new_text_joined)
    print("\n\n")
    new_text_token_ids = tokenizer.encode(new_text_joined)
    print(new_text_token_ids)

    print("Back to text:\n")
    print(tokenizer.decode(new_text_token_ids))