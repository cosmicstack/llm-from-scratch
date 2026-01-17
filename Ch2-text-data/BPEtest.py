from importlib.metadata import version
import tiktoken

print("tiktoken version: {}".format(version("tiktoken")))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)


# Let's try some unknown text on the GPT-2 BPE tokenizer
print("=========== UNK TRIALS ============= \n\n")
unk_text = "ennamma kannu sowkiyamma?"
ints = tokenizer.encode(unk_text)
print(ints)
print("\n\n")
print(tokenizer.decode(ints))

print("\n\n")

for i in range(len(ints)):
    print(tokenizer.decode([ints[i]]))