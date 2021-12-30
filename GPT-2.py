from aitextgen import aitextgen
from aitextgen.colab import mount_gdrive, copy_file_from_gdrive
import logging
import tensorflow
from pkg_resources import resource_filename
import yaml
import glob
from transformers import GPT2Tokenizer, GPT2Model
import torch
from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen


def main():
    # hf_model = "minimaxir/hacker-news"
    hf_model = "EleutherAI/gpt-neo-125M"
    ai = aitextgen(model=hf_model, verbose=True)
    ai.generate_to_file(n=10, prompt="Twitter", max_length=100, temperature=1.2)


if __name__ == '__main__':
    main()
