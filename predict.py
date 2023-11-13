import time
from typing import Optional
import subprocess

import torch
import os

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from collections import OrderedDict
from cog import BasePredictor, ConcatenateIterator, Input, Path

# from config import DEFAULT_MODEL_NAME, DEFAULT_CONFIG_PATH, load_tokenizer, load_tensorizer
from subclass import YieldingReplitCode

# Weights are either local or in a cloud bucket.

# For development, point to a local path on disk.
# This is the path from which we pull weights when there's no COG_WEIGHTS environment variable (COG_WEIGHTS is a thing for trainable models)
# TENSORIZER_WEIGHTS_PATH = "model/model.tensors"
TENSORIZER_WEIGHTS_URL = "https://weights.replicate.delivery/default/replit-code-v1-3b/model.tensors"
TENSORIZER_WEIGHTS_PATH = "model/model.tensors"

# Set this to a GCP URL when pushing the model
# TENSORIZER_WEIGHTS_PATH = None

DEFAULT_CONFIG_PATH = "model/"
TOKENIZER_PATH = "model/"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # set TOKENIZERS_PARALLELISM to false to avoid a warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not os.path.exists(TENSORIZER_WEIGHTS_PATH):
            download_weights(TENSORIZER_WEIGHTS_URL, TENSORIZER_WEIGHTS_PATH)

        self.model = self.load_tensorizer(
            weights=TENSORIZER_WEIGHTS_PATH, plaid_mode=True, cls=YieldingReplitCode, config_path=DEFAULT_CONFIG_PATH,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    def load_tensorizer(self, weights, plaid_mode, cls, config_path):
        st = time.time()
        print(f"deserializing weights from {weights}")

        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        config.attn_config['attn_impl'] = 'triton'

        # with no_init_or_tensor():
        #     model = YieldingReplitCode.from_pretrained('./model/', config=config, trust_remote_code=True)


        model = no_init_or_tensor(
            lambda: cls.from_pretrained(
                None, config=config, state_dict=OrderedDict(), trust_remote_code=True,
            )
        )


        deserialized = TensorDeserializer(weights, plaid_mode=True)
        deserialized.load_into_module(model)
        try:
          model = model.to(dtype=torch.bfloat16)
        except:
            pass

        print(f"weights loaded in {time.time() - st}")
        return model

    def predict(
        self,
        prompt: str = Input(description=f"Text prompt"),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=500,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1,
        ),
        length_penalty: float = Input(
            description="Increasing the length_penalty parameter above 1.0 will cause the model to favor longer sequences, while decreasing it below 1.0 will cause the model to favor shorter sequences.",
            ge=0.01,
            le=5,
            default=1,
        ),
        no_repeat_ngram_size: int = Input(
            description="If set to int > 0, all ngrams of size no_repeat_ngram_size can only occur once.",
            ge=0,
            default=0,
        ),
        stop_sequence: str = Input(
            description="Generation will hault if this token is produced. Currently, only single token stop sequences are support and it is recommended to use `###` as the stop sequence if you want to control generation termination.",
            default=None,
        ),
        seed: int = Input(
            description="Set seed for reproducible outputs. Set to -1 for random seed.",
            ge=-1,
            default=-1,
        ),
        debug: bool = Input(
            description="provide debugging output in logs", default=False
        ),
    ) -> ConcatenateIterator[str]:
        input = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # set torch seed
        if seed == -1:
            torch.seed()

        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        with torch.inference_mode():
            first_token_yielded = False
            prev_ids = []
            for output in self.model.generate(
                input,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            ):
                cur_id = output.item()

                # in order to properly handle spaces, we need to do our own tokenizing. Fun!
                # we're building up a buffer of sub-word / punctuation tokens until we hit a space, and then yielding whole words + punctuation.
                cur_token = self.tokenizer.convert_ids_to_tokens(cur_id)

                # skip initial newline, which this almost always yields. hack - newline id = 13.
                if not first_token_yielded and not prev_ids and cur_id == 187:
                    continue

                # Ġ means a space, means we yield previous tokens
                if cur_token.startswith("Ġ"):  # this is not a standard G.
                    # first token
                    if not prev_ids:
                        prev_ids = [cur_id]
                        continue

                    # there are tokens to yield
                    else:
                        token = self.tokenizer.decode(prev_ids, clean_up_tokenization_spaces=False)
                        prev_ids = [cur_id]

                        if not first_token_yielded:
                            # no leading space for first token
                            token = token.strip()
                            first_token_yielded = True
                        yield token
                                # End token
                elif cur_token == "<|endoftext|>":
                    break

                elif stop_sequence and cur_token == stop_sequence:
                    break

                else:
                    prev_ids.append(cur_id)
                    continue

            # remove any special tokens such as </s>
            token = self.tokenizer.decode(prev_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if not first_token_yielded:
                # no leading space for first token
                token = token.strip()
                first_token_yielded = True
            yield token

        if debug:
            print(f"cur memory: {torch.cuda.memory_allocated()}")
            print(f"max allocated: {torch.cuda.max_memory_allocated()}")
            print(f"peak memory: {torch.cuda.max_memory_reserved()}")
