# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional

from requests.exceptions import HTTPError


def hf_download(repo_id: Optional[str] = None, hf_token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download
    os.makedirs(f"checkpoints/{repo_id}", exist_ok=True)
    
    # Set allow_patterns to ensure we download tokenizer files for models like LLaDA
    # that use the HuggingFace tokenizer format
    allow_patterns = [
        "*.json",          # Config files and tokenizer files
        "*.bin",           # Model weights
        "*.safetensors",   # Model weights in safetensors format
        "*.model",         # SentencePiece tokenizer models
        "*.py",            # Model code
        "*.txt",           # Readme and other text files
        "*.md",            # Markdown documentation  
        "tokenizer.*",     # Any tokenizer files
        "vocab.*",         # Vocabulary files
        "merges.txt",      # BPE merges
        "special_tokens_map.json",  # Special tokens mapping
        "tokenizer_config.json",    # Tokenizer config
    ]
    
    try:
        snapshot_download(
            repo_id, 
            local_dir=f"checkpoints/{repo_id}", 
            local_dir_use_symlinks=False, 
            token=hf_token,
            allow_patterns=allow_patterns
        )
        print(f"Downloaded model from {repo_id} to checkpoints/{repo_id}")
    except HTTPError as e:
        if e.response.status_code == 401:
            print("You need to pass a valid `--hf_token=...` to download private checkpoints.")
        else:
            raise e

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download data from HuggingFace Hub.')
    parser.add_argument('--repo_id', type=str, default="checkpoints/meta-llama/llama-2-7b-chat-hf", help='Repository ID to download from.')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace API token.')

    args = parser.parse_args()
    hf_download(args.repo_id, args.hf_token)
