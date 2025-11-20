from huggingface_hub import hf_hub_download
import json

# Download the config.json from the model
config_path = hf_hub_download(repo_id="deepseek-ai/DeepSeek-OCR", filename="config.json")

# Read and print the config
with open(config_path, 'r') as f:
    config = json.load(f)
    print(json.dumps(config, indent=2))
