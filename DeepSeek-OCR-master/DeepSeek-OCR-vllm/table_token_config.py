"""
Configuration for table-related token IDs and improved settings for wide table processing.

This module provides:
1. Extended whitelist of table-related tokens for NoRepeatNGramLogitsProcessor
2. Optimized parameters for processing documents with wide tables
3. Helper functions to adjust settings based on document complexity
"""

# Known table-related token IDs
# Based on the codebase, we know:
# - 128821: <td>
# - 128822: </td>
# 
# We need to find the IDs for: <tr>, </tr>, <table>, </table>, <thead>, </thead>, <tbody>, </tbody>, <th>, </th>
# These are typically sequential or nearby in the vocabulary

# Extended whitelist for table-related tokens
# This includes all HTML table structure tokens that should be allowed to repeat
TABLE_TOKEN_WHITELIST = {
    128821,  # <td>
    128822,  # </td>
    128823,  # <tr> (estimated - needs verification)
    128824,  # </tr> (estimated - needs verification)
    # Add more as discovered
}

# Conservative whitelist (confirmed tokens only)
CONFIRMED_TABLE_TOKENS = {
    128821,  # <td>
    128822,  # </td>
}

# Function to get table token IDs from tokenizer
def get_table_token_ids(tokenizer):
    """
    Dynamically find all table-related token IDs from the tokenizer.
    
    Args:
        tokenizer: The tokenizer instance
        
    Returns:
        set: Set of token IDs for table-related tokens
    """
    table_tokens = [
        '<td>', '</td>',
        '<tr>', '</tr>',
        '<table>', '</table>',
        '<thead>', '</thead>',
        '<tbody>', '</tbody>',
        '<th>', '</th>',
    ]
    
    token_ids = set()
    for token in table_tokens:
        token_id = tokenizer.vocab.get(token)
        if token_id is not None:
            token_ids.add(token_id)
    
    # Always include the confirmed tokens
    token_ids.update(CONFIRMED_TABLE_TOKENS)
    
    return token_ids


# Optimized parameters for different document types
class TableProcessingConfig:
    """Configuration presets for different document types."""
    
    # Default settings (original)
    DEFAULT = {
        'ngram_size': 30,
        'window_size': 90,
        'max_tokens': 8192,
        'max_model_len': 8192,
        'whitelist_token_ids': CONFIRMED_TABLE_TOKENS,
    }
    
    # Optimized for documents with wide tables
    WIDE_TABLE = {
        'ngram_size': 20,  # More lenient to allow table repetition
        'window_size': 70,  # Smaller window for faster processing
        'max_tokens': 12288,  # Increased for wide tables
        'max_model_len': 12288,
        'whitelist_token_ids': TABLE_TOKEN_WHITELIST,
    }
    
    # Optimized for documents with many tables
    TABLE_HEAVY = {
        'ngram_size': 25,
        'window_size': 80,
        'max_tokens': 10240,
        'max_model_len': 10240,
        'whitelist_token_ids': TABLE_TOKEN_WHITELIST,
    }
    
    # For PDF processing (faster, more lenient)
    PDF_OPTIMIZED = {
        'ngram_size': 15,  # Very lenient for multi-page PDFs
        'window_size': 50,
        'max_tokens': 12288,
        'max_model_len': 12288,
        'whitelist_token_ids': TABLE_TOKEN_WHITELIST,
    }
    
    # For batch evaluation (balanced)
    BATCH_EVAL = {
        'ngram_size': 25,
        'window_size': 75,
        'max_tokens': 10240,
        'max_model_len': 10240,
        'whitelist_token_ids': TABLE_TOKEN_WHITELIST,
    }


def get_config_for_document_type(doc_type='default'):
    """
    Get optimized configuration for a specific document type.
    
    Args:
        doc_type (str): One of 'default', 'wide_table', 'table_heavy', 'pdf', 'batch_eval'
        
    Returns:
        dict: Configuration dictionary
    """
    configs = {
        'default': TableProcessingConfig.DEFAULT,
        'wide_table': TableProcessingConfig.WIDE_TABLE,
        'table_heavy': TableProcessingConfig.TABLE_HEAVY,
        'pdf': TableProcessingConfig.PDF_OPTIMIZED,
        'batch_eval': TableProcessingConfig.BATCH_EVAL,
    }
    
    return configs.get(doc_type, TableProcessingConfig.DEFAULT)


def create_logits_processor(config_dict, tokenizer=None):
    """
    Create a NoRepeatNGramLogitsProcessor with the given configuration.
    
    Args:
        config_dict (dict): Configuration dictionary
        tokenizer: Optional tokenizer to dynamically find table tokens
        
    Returns:
        NoRepeatNGramLogitsProcessor instance
    """
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
    
    whitelist = config_dict['whitelist_token_ids']
    
    # If tokenizer is provided, dynamically find table tokens
    if tokenizer is not None:
        whitelist = get_table_token_ids(tokenizer)
    
    return NoRepeatNGramLogitsProcessor(
        ngram_size=config_dict['ngram_size'],
        window_size=config_dict['window_size'],
        whitelist_token_ids=whitelist
    )


# Helper function to detect if a document likely contains wide tables
def estimate_table_complexity(image_width, image_height, aspect_ratio_threshold=2.0):
    """
    Estimate if a document likely contains wide tables based on dimensions.
    
    Args:
        image_width (int): Image width in pixels
        image_height (int): Image height in pixels
        aspect_ratio_threshold (float): Threshold for considering an image "wide"
        
    Returns:
        str: Suggested config type ('wide_table', 'table_heavy', or 'default')
    """
    aspect_ratio = image_width / image_height
    
    # Very wide images likely have wide tables
    if aspect_ratio > aspect_ratio_threshold:
        return 'wide_table'
    
    # Large images might have complex tables
    if image_width > 2000 or image_height > 2000:
        return 'table_heavy'
    
    return 'default'


# Monitoring helper
def check_token_usage(output_length, max_tokens, warning_threshold=0.9):
    """
    Check if token usage is approaching the limit and print a warning.
    
    Args:
        output_length (int): Current output length in tokens
        max_tokens (int): Maximum allowed tokens
        warning_threshold (float): Threshold (0-1) for warning
        
    Returns:
        bool: True if approaching limit, False otherwise
    """
    usage_ratio = output_length / max_tokens
    
    if usage_ratio >= warning_threshold:
        print(f"\n⚠️  WARNING: Output approaching token limit!")
        print(f"   Current: {output_length} / {max_tokens} tokens ({usage_ratio*100:.1f}%)")
        print(f"   Consider increasing max_tokens or using a more lenient configuration.")
        return True
    
    return False
