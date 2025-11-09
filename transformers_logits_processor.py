"""
NoRepeatNGramLogitsProcessor for Transformers
Prevents hallucination by blocking repetitive n-grams during generation.

This is adapted from the vLLM implementation to work with HuggingFace Transformers.
"""

import torch
from transformers import LogitsProcessor
from typing import List, Set, Optional


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    Logits processor that prevents the generation of repeated n-grams.
    
    This processor looks at the last (ngram_size - 1) tokens and checks if they
    have appeared before in a sliding window. If they have, it bans the tokens
    that would complete those n-grams, preventing repetitive patterns.
    
    Args:
        ngram_size (int): The size of n-grams to check for repetition.
            Larger values (30-40) are better for preventing long repetitive patterns.
        window_size (int): The size of the sliding window to check for n-grams.
            Larger values (90-120) check more history but are slower.
        whitelist_token_ids (set, optional): Token IDs that are allowed to repeat.
            Useful for structural tokens like <td>, </td> in tables.
    
    Example:
        >>> processor = NoRepeatNGramLogitsProcessor(
        ...     ngram_size=30,
        ...     window_size=90,
        ...     whitelist_token_ids={128821, 128822}  # <td>, </td>
        ... )
        >>> logits_processors = [processor]
        >>> outputs = model.generate(
        ...     input_ids,
        ...     logits_processor=logits_processors,
        ...     max_new_tokens=8192,
        ...     temperature=0.0
        ... )
    """
    
    def __init__(
        self, 
        ngram_size: int, 
        window_size: int = 100, 
        whitelist_token_ids: Optional[Set[int]] = None
    ):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}"
            )
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(
                f"`window_size` has to be a strictly positive integer, but is {window_size}"
            )
        
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()
    
    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Process logits to prevent n-gram repetition.
        
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length) containing input token IDs
            scores: Tensor of shape (batch_size, vocab_size) containing logits for next token
            
        Returns:
            Modified scores with banned tokens set to -inf
        """
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[-1]
        
        # Process each sequence in the batch
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx].tolist()
            
            # Need at least ngram_size tokens to check for repetition
            if len(sequence) < self.ngram_size:
                continue
            
            # Get the current prefix (last ngram_size - 1 tokens)
            current_prefix = tuple(sequence[-(self.ngram_size - 1):])
            
            # Define the search window
            search_start = max(0, len(sequence) - self.window_size)
            search_end = len(sequence) - self.ngram_size + 1
            
            # Find all tokens that would complete a repeated n-gram
            banned_tokens = set()
            for i in range(search_start, search_end):
                ngram = tuple(sequence[i:i + self.ngram_size])
                # If the prefix matches, ban the completing token
                if ngram[:-1] == current_prefix:
                    banned_tokens.add(ngram[-1])
            
            # Remove whitelisted tokens from banned set
            banned_tokens = banned_tokens - self.whitelist_token_ids
            
            # Apply bans by setting logits to -inf
            if banned_tokens:
                for token in banned_tokens:
                    if token < vocab_size:  # Safety check
                        scores[batch_idx, token] = float("-inf")
        
        return scores


class AdaptiveNoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    Advanced version that adapts ngram_size based on generation progress.
    
    Uses larger n-grams early in generation (when context is limited) and
    smaller n-grams later (to allow more flexibility while still preventing loops).
    
    Args:
        initial_ngram_size (int): N-gram size at the start of generation
        final_ngram_size (int): N-gram size after transition_tokens
        transition_tokens (int): Number of tokens after which to use final_ngram_size
        window_size (int): Size of the sliding window
        whitelist_token_ids (set, optional): Token IDs allowed to repeat
    """
    
    def __init__(
        self,
        initial_ngram_size: int = 40,
        final_ngram_size: int = 20,
        transition_tokens: int = 1000,
        window_size: int = 100,
        whitelist_token_ids: Optional[Set[int]] = None
    ):
        self.initial_ngram_size = initial_ngram_size
        self.final_ngram_size = final_ngram_size
        self.transition_tokens = transition_tokens
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()
        self.start_length = None
    
    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[-1]
        
        # Initialize start length on first call
        if self.start_length is None:
            self.start_length = input_ids.shape[1]
        
        # Calculate current ngram size based on generation progress
        tokens_generated = input_ids.shape[1] - self.start_length
        if tokens_generated < self.transition_tokens:
            # Linear interpolation between initial and final
            progress = tokens_generated / self.transition_tokens
            current_ngram_size = int(
                self.initial_ngram_size - 
                (self.initial_ngram_size - self.final_ngram_size) * progress
            )
        else:
            current_ngram_size = self.final_ngram_size
        
        # Process each sequence in the batch
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx].tolist()
            
            if len(sequence) < current_ngram_size:
                continue
            
            current_prefix = tuple(sequence[-(current_ngram_size - 1):])
            search_start = max(0, len(sequence) - self.window_size)
            search_end = len(sequence) - current_ngram_size + 1
            
            banned_tokens = set()
            for i in range(search_start, search_end):
                ngram = tuple(sequence[i:i + current_ngram_size])
                if ngram[:-1] == current_prefix:
                    banned_tokens.add(ngram[-1])
            
            banned_tokens = banned_tokens - self.whitelist_token_ids
            
            if banned_tokens:
                for token in banned_tokens:
                    if token < vocab_size:
                        scores[batch_idx, token] = float("-inf")
        
        return scores


# Convenience function for easy usage
def create_anti_hallucination_processors(
    mode: str = "standard",
    ngram_size: int = 30,
    window_size: int = 90,
    whitelist_token_ids: Optional[Set[int]] = None
) -> List[LogitsProcessor]:
    """
    Create a list of logits processors to prevent hallucination.
    
    Args:
        mode: "standard" for fixed n-gram size, "adaptive" for dynamic sizing
        ngram_size: Size of n-grams to block (for standard mode)
        window_size: Size of sliding window to check
        whitelist_token_ids: Token IDs allowed to repeat (e.g., {128821, 128822} for <td>, </td>)
    
    Returns:
        List of LogitsProcessor instances
    
    Example:
        >>> processors = create_anti_hallucination_processors(
        ...     mode="standard",
        ...     ngram_size=30,
        ...     window_size=90,
        ...     whitelist_token_ids={128821, 128822}
        ... )
        >>> outputs = model.generate(
        ...     input_ids,
        ...     logits_processor=processors,
        ...     max_new_tokens=8192
        ... )
    """
    if whitelist_token_ids is None:
        # Default whitelist for table tokens
        whitelist_token_ids = {128821, 128822}  # <td>, </td>
    
    if mode == "standard":
        return [NoRepeatNGramLogitsProcessor(
            ngram_size=ngram_size,
            window_size=window_size,
            whitelist_token_ids=whitelist_token_ids
        )]
    elif mode == "adaptive":
        return [AdaptiveNoRepeatNGramLogitsProcessor(
            initial_ngram_size=ngram_size + 10,
            final_ngram_size=max(20, ngram_size - 10),
            transition_tokens=1000,
            window_size=window_size,
            whitelist_token_ids=whitelist_token_ids
        )]
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'standard' or 'adaptive'")
