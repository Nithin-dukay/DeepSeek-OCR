from transformers import AutoTokenizer

MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# The whitelist token ids from the README
whitelist = {128821, 128822}

for token_id in whitelist:
    token = tokenizer.decode(token_id)
    print(f"Token ID {token_id}: '{token}'")

# Also check some common tokens
print("\nCommon tokens:")
print(f"Space: {tokenizer.encode(' ')}")
print(f"Newline: {tokenizer.encode('\n')}")
print(f"Digit 0: {tokenizer.encode('0')}")
print(f"Digit 1: {tokenizer.encode('1')}")

# Check if numbers are in vocab
for i in range(10):
    token_id = tokenizer.encode(str(i))[0]
    print(f"Digit {i}: {token_id}")