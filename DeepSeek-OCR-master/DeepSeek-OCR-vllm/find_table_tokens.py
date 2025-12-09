"""
Utility script to find token IDs for table-related special tokens.
This helps identify which tokens should be whitelisted in the NoRepeatNGramLogitsProcessor.
"""

from transformers import AutoTokenizer
import sys

def find_table_token_ids(model_path='deepseek-ai/DeepSeek-OCR'):
    """Find and print token IDs for all table-related special tokens."""
    
    print(f"Loading tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to local config...")
        try:
            from config import TOKENIZER
            tokenizer = TOKENIZER
        except:
            print("Could not load tokenizer. Please ensure model is downloaded or TOKENIZER is configured.")
            return None
    
    # List of table-related tokens to check
    table_tokens = [
        '<td>',
        '</td>',
        '<tr>',
        '</tr>',
        '<table>',
        '</table>',
        '<thead>',
        '</thead>',
        '<tbody>',
        '</tbody>',
        '<th>',
        '</th>',
    ]
    
    print("\n" + "="*60)
    print("TABLE-RELATED TOKEN IDs")
    print("="*60)
    
    found_tokens = {}
    missing_tokens = []
    
    for token in table_tokens:
        token_id = tokenizer.vocab.get(token)
        if token_id is not None:
            found_tokens[token] = token_id
            print(f"{token:15s} -> {token_id}")
        else:
            missing_tokens.append(token)
            print(f"{token:15s} -> NOT FOUND")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Found {len(found_tokens)} tokens")
    print(f"Missing {len(missing_tokens)} tokens")
    
    if found_tokens:
        print("\n" + "="*60)
        print("WHITELIST SET FOR CODE")
        print("="*60)
        token_ids = sorted(found_tokens.values())
        print(f"whitelist_token_ids = {{{', '.join(map(str, token_ids))}}}")
        print("\n# Token mapping:")
        for token, token_id in sorted(found_tokens.items(), key=lambda x: x[1]):
            print(f"# {token_id}: {token}")
    
    if missing_tokens:
        print("\n" + "="*60)
        print("MISSING TOKENS (not in vocabulary)")
        print("="*60)
        for token in missing_tokens:
            print(f"  - {token}")
    
    return found_tokens

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'deepseek-ai/DeepSeek-OCR'
    find_table_token_ids(model_path)
