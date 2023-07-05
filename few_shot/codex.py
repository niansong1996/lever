import openai
import os
import time 
import json
import numpy as np

from tqdm import tqdm
from typing import List, Optional, Dict, Any, Callable, Tuple
# from execution.execution_evaluation import batch_eval_at_k

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# OPENAI_API_ORG = os.environ.get('OPENAI_API_ORG')
if OPENAI_API_KEY is None:
    raise Exception("Please set your OpenAI API key in the environment variable OPENAI_API_KEY")
else:
    openai.api_key = OPENAI_API_KEY
    # openai.organization = OPENAI_API_ORG

# From the Codex Paper
def estimate_pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k: 
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def openai_call(prompts: List[str], engine: str, max_tokens: int = 1024, temperature: Optional[float] = None, 
             top_p: Optional[float] = None, n: Optional[int] = None, best_of: Optional[int] = None, 
             wait_on_limit: bool = True, stop: str = None, 
             get_raw_generation_results: bool = False, **kwargs) -> List[List[str]]:
    """
    Takes a list of prompts and returns a list of generated responses for each prompt.
    
    For more arguments to https://beta.openai.com/docs/api-reference/completions/create
    """
    # prune out the None arguments since openAI api treats unspecified arguments differently
    def prune_none_args(**kwargs):
        return {k: v for k, v in kwargs.items() if v is not None}
    
    if get_raw_generation_results:
        kwargs['logprobs'] = 1

    while True:
        try:
            non_none_args = prune_none_args(engine=engine, prompt=prompts, max_tokens=max_tokens, 
                                            temperature=temperature, top_p=top_p, n=n, best_of=best_of, 
                                            stop=stop, **kwargs)
            completion = openai.Completion.create(**non_none_args)
            break
        except openai.error.RateLimitError as e:
            print(e)
            if wait_on_limit:
                print("RateLimitError occurred, waiting for 30 seconds and retrying...")
                time.sleep(30)
            else:
                raise e
        except openai.error.APIError as e:
            print("APIError occurred, retry in 5 seconds...")
            time.sleep(5)
        except openai.error.ServiceUnavailableError as e:
            print("openai.error.ServiceUnavailableError occurred, retry in 30 seconds...")
            time.sleep(30)
        except openai.error.APIConnectionError as e:
            print("openai.error.APIConnectionError occurred, retry in 30 seconds...")
            time.sleep(30)
        except Exception as e:
            print("Other unknown exception occured, retry in 5 mins...")
            time.sleep(60 * 5)

    # get the text from the returned results and slice the completions to input_n * completion_n
    completion_texts = [x.text for x in completion.choices]
    completion_results = [completion_texts[i*n:(i+1)*n] for i in range(len(prompts))] 

    if get_raw_generation_results:
        logprobs = [x.logprobs.token_logprobs for x in completion.choices]
        gen_tokens = [x.logprobs.tokens for x in completion.choices]
        logprobs_results = [logprobs[i*n:(i+1)*n] for i in range(len(prompts))]
        gen_tokens_results = [gen_tokens[i*n:(i+1)*n] for i in range(len(prompts))]
        return completion_results, logprobs_results, gen_tokens_results
    else:
        return completion_results