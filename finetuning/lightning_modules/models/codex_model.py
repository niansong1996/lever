import torch

from typing import Dict, List, Any, Optional, Tuple

from itertools import chain

from transformers.generation_utils import GenerationMixin
from transformers import PreTrainedTokenizer 

from few_shot.codex import openai_call


class CodexModel(GenerationMixin):
    def __init__(self, 
                 engine: str,
                 tokenizer: PreTrainedTokenizer,
                 stop_seq: str = None,
                 **kwargs
                 ) -> None:
        assert engine in ["code-davinci-002", "code-cushman-001", "code-davinci-001"], "CodexModel only supports code-davinci-001/002 and code-cushman-001"
        self.engine = engine
        self.stop_seq = stop_seq
        # use this tokenizer to decode the tokenized input, but the output will be string
        self.tokenizer = tokenizer

        super().__init__(**kwargs)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError("CodexModel is not trainable")
    
    def generate(self, 
                 input_ids: torch.Tensor, 
                 max_new_tokens: int,
                 temperature: float,
                 do_sample: bool = True,
                 num_return_sequences: int = 1, 
                 attention_mask: torch.Tensor = None, # must be put here to match the signature of generate
                 return_dict_in_generate: bool = True,
                 output_scores: bool = True,
                 num_beams: int = 1,
                 ) -> List[str]:

        assert do_sample, "CodexModel only supports do_sample=True"
        assert num_beams == 1, "CodexModel only supports num_beams=1"
        num_return_sequences = 1 if num_return_sequences is None else num_return_sequences # to deal with default hf values

        # init the query args and prompts
        openai_kwargs = {'engine': self.engine, 'max_tokens': max_new_tokens, 'n': num_return_sequences, 
                         'temperature': temperature, 'stop': self.stop_seq, 'get_raw_generation_results': True}
        input_prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # call the openai api and chain the results
        openai_results, logprobs, gen_tokens = openai_call(input_prompts, **openai_kwargs)
        flatten_results = list(chain.from_iterable(openai_results))
        flatten_logprobs = list(chain.from_iterable(logprobs))
        flatten_gen_tokens = list(chain.from_iterable(gen_tokens))

        return {"sequences": flatten_results, "scores": flatten_logprobs, "tokens": flatten_gen_tokens}
