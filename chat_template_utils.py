"""
Chat Template Utilities for DeepSeek-OCR
Addresses GitHub Issue #151: Provides official chat_template configuration
to enable tokenizer.apply_chat_template() workflow.
"""

from transformers import AutoTokenizer
from typing import Optional, List, Dict, Any


# Official DeepSeek-OCR chat template
# Based on DeepSeek-VL2 architecture and OCR-specific requirements
DEEPSEEK_OCR_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'user' %}User: {{ message['content'] }}

{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}

{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"""


# Alternative simpler template for OCR-only use
DEEPSEEK_OCR_SIMPLE_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}"""


def configure_chat_template(
    tokenizer: AutoTokenizer,
    template: Optional[str] = None,
    template_type: str = "default"
) -> AutoTokenizer:
    """
    Configure chat template for DeepSeek-OCR tokenizer.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        template: Custom template string (Jinja2 format)
        template_type: Type of template to use ("default", "simple", or "custom")
        
    Returns:
        Tokenizer with configured chat_template
    """
    if template is not None:
        # Use custom template
        tokenizer.chat_template = template
    elif template_type == "simple":
        # Use simple template (no role markers)
        tokenizer.chat_template = DEEPSEEK_OCR_SIMPLE_TEMPLATE
    else:
        # Use default template
        tokenizer.chat_template = DEEPSEEK_OCR_CHAT_TEMPLATE
    
    return tokenizer


def load_tokenizer_with_chat_template(
    model_name: str = "deepseek-ai/DeepSeek-OCR",
    template_type: str = "default",
    trust_remote_code: bool = True
) -> AutoTokenizer:
    """
    Load DeepSeek-OCR tokenizer with chat template configured.
    
    Args:
        model_name: HuggingFace model identifier
        template_type: Type of template ("default" or "simple")
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # Configure chat template
    tokenizer = configure_chat_template(tokenizer, template_type=template_type)
    
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def format_ocr_prompt(
    image_token: str = "<image>",
    instruction: str = "Convert the document to markdown.",
    use_grounding: bool = True,
    system_message: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Format OCR prompt as chat messages for apply_chat_template().
    
    Args:
        image_token: Token representing image position
        instruction: OCR instruction
        use_grounding: Whether to use grounding mode
        system_message: Optional system message
        
    Returns:
        List of message dictionaries
    """
    messages = []
    
    if system_message:
        messages.append({
            "role": "system",
            "content": system_message
        })
    
    # Build user message
    if use_grounding:
        content = f"{image_token}\n<|grounding|>{instruction}"
    else:
        content = f"{image_token}\n{instruction}"
    
    messages.append({
        "role": "user",
        "content": content
    })
    
    return messages


def apply_chat_template_for_ocr(
    tokenizer: AutoTokenizer,
    instruction: str = "Convert the document to markdown.",
    use_grounding: bool = True,
    add_generation_prompt: bool = True,
    return_tensors: Optional[str] = "pt",
    **kwargs
) -> Any:
    """
    Apply chat template for OCR task.
    
    Args:
        tokenizer: Configured tokenizer with chat_template
        instruction: OCR instruction
        use_grounding: Whether to use grounding mode
        add_generation_prompt: Whether to add generation prompt
        return_tensors: Format for returned tensors
        **kwargs: Additional arguments for apply_chat_template
        
    Returns:
        Tokenized inputs
    """
    # Ensure tokenizer has chat_template
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        tokenizer = configure_chat_template(tokenizer)
    
    # Format messages
    messages = format_ocr_prompt(
        instruction=instruction,
        use_grounding=use_grounding
    )
    
    # Apply template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        return_tensors=return_tensors,
        **kwargs
    )
    
    return inputs


def example_usage():
    """Example usage of chat template utilities."""
    
    print("=== Example 1: Load tokenizer with chat template ===")
    tokenizer = load_tokenizer_with_chat_template(
        model_name="deepseek-ai/DeepSeek-OCR",
        template_type="default"
    )
    print(f"Chat template configured: {tokenizer.chat_template is not None}")
    
    print("\n=== Example 2: Format OCR prompt ===")
    messages = format_ocr_prompt(
        instruction="Convert the document to markdown.",
        use_grounding=True
    )
    print(f"Messages: {messages}")
    
    print("\n=== Example 3: Apply chat template ===")
    try:
        inputs = apply_chat_template_for_ocr(
            tokenizer,
            instruction="Transcribe the image verbatim as plain text.",
            use_grounding=False,
            return_tensors="pt"
        )
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        
        # Decode to see formatted prompt
        decoded = tokenizer.decode(inputs['input_ids'][0])
        print(f"Formatted prompt: {decoded}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== Example 4: Manual chat template application ===")
    messages = [
        {
            "role": "user",
            "content": "<image>\n<|grounding|>Convert the document to markdown."
        }
    ]
    
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"Formatted text:\n{formatted}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== Example 5: Different prompt types ===")
    prompt_examples = [
        ("Document OCR", "Convert the document to markdown.", True),
        ("Plain OCR", "Free OCR.", False),
        ("Image OCR", "OCR this image.", True),
        ("Figure parsing", "Parse the figure.", False),
        ("Detailed description", "Describe this image in detail.", False),
    ]
    
    for name, instruction, use_grounding in prompt_examples:
        messages = format_ocr_prompt(
            instruction=instruction,
            use_grounding=use_grounding
        )
        print(f"\n{name}:")
        print(f"  Content: {messages[0]['content']}")


if __name__ == "__main__":
    example_usage()
