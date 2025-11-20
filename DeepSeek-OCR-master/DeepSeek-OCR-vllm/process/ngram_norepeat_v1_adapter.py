from .ngram_norepeat import NoRepeatNGramLogitsProcessor
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


class NoRepeatNGramAdaptor(AdapterLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return True

    def new_req_logits_processor(self, params):
        return NoRepeatNGramLogitsProcessor(
            ngram_size=params.extra_args["ngram_size"],
            window_size=params.extra_args["window_size"],
            whitelist_token_ids=params.extra_args["whitelist_token_ids"],
        )