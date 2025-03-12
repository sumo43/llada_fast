import os
import sentencepiece as spm
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path
from typing import Dict, Optional, List, Union
import json

# Import transformers if available, otherwise we'll handle the ImportError when needed
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class TokenizerInterface:
    def __init__(self, model_path):
        self.model_path = model_path

    def encode(self, text):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode(self, tokens):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def bos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def eos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

class SentencePieceWrapper(TokenizerInterface):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.processor = spm.SentencePieceProcessor(str(model_path))

    def encode(self, text):
        return self.processor.EncodeAsIds(text)

    def decode(self, tokens):
        return self.processor.DecodeIds(tokens)

    def bos_id(self):
        return self.processor.bos_id()

    def eos_id(self):
        return self.processor.eos_id()

class TiktokenWrapper(TokenizerInterface):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path):
        super().__init__(model_path)
        assert os.path.isfile(model_path), str(model_path)
        mergeable_ranks = load_tiktoken_bpe(str(model_path))
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        # BOS / EOS token IDs
        self._bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self._eos_id: int = self.special_tokens["<|end_of_text|>"]

    def encode(self, text):
        return self.model.encode(text)

    def decode(self, tokens):
        return self.model.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id

class HuggingFaceTokenizerWrapper(TokenizerInterface):
    """
    Tokenizing and encoding/decoding text using the HuggingFace tokenizer.
    """
    
    def __init__(self, model_path_or_dir):
        """
        Initialize with either a model file path or a directory containing the tokenizer files.
        
        Args:
            model_path_or_dir: Path to tokenizer model file or directory containing tokenizer files
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("The 'transformers' library is required for HuggingFaceTokenizerWrapper. "
                             "Please install it with 'pip install transformers'.")
        
        super().__init__(model_path_or_dir)
        
        # Check if path is directory or file
        path = Path(model_path_or_dir)
        if path.is_dir():
            # Load from directory
            self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        else:
            # Try to determine the parent directory
            parent_dir = path.parent
            self.tokenizer = AutoTokenizer.from_pretrained(str(parent_dir))
            
        # Load config to get BOS/EOS tokens
        config_path = Path(path.parent) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                self._bos_id = config.get("bos_token_id", None)
                self._eos_id = config.get("eos_token_id", None)
                
        # If not found in config, use tokenizer defaults
        if getattr(self, "_bos_id", None) is None and hasattr(self.tokenizer, "bos_token_id"):
            self._bos_id = self.tokenizer.bos_token_id
            
        if getattr(self, "_eos_id", None) is None and hasattr(self.tokenizer, "eos_token_id"):
            self._eos_id = self.tokenizer.eos_token_id
            
        # Fallback values if still not set
        if getattr(self, "_bos_id", None) is None:
            self._bos_id = 1
        if getattr(self, "_eos_id", None) is None:
            self._eos_id = 2

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        if isinstance(tokens, (list, tuple)) and len(tokens) == 0:
            return ""
        return self.tokenizer.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id

def get_tokenizer(tokenizer_model_path, model_name):
    """
    Factory function to get the appropriate tokenizer based on the model name.
    
    Args:
    - tokenizer_model_path (str): The file path to the tokenizer model or directory.
    - model_name (str): The name of the model, used to determine the tokenizer type.

    Returns:
    - TokenizerInterface: An instance of a tokenizer.
    """
    path = Path(tokenizer_model_path)
    
    # Try to detect if we have a HuggingFace tokenizer
    if "LLaDA" in str(model_name):
        # For LLaDA models, use the HuggingFace tokenizer
        if TRANSFORMERS_AVAILABLE:
            # If path is a directory, check for HF tokenizer files in it
            if path.is_dir():
                hf_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
                if any((path / file).exists() for file in hf_files):
                    print(f"Using HuggingFace tokenizer from directory: {path}")
                    return HuggingFaceTokenizerWrapper(path)
            # If path is a file, check parent directory
            elif path.is_file():
                parent_dir = path.parent
                hf_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
                if any((parent_dir / file).exists() for file in hf_files):
                    print(f"Using HuggingFace tokenizer from directory: {parent_dir}")
                    return HuggingFaceTokenizerWrapper(parent_dir)
    
    # Use tiktoken for Llama 3 models
    if "llama-3" in str(model_name).lower():
        if path.is_file():
            return TiktokenWrapper(tokenizer_model_path)
        else:
            raise ValueError(f"For Llama 3 models, tokenizer_model_path must be a file, got: {tokenizer_model_path}")
    
    # Default to SentencePiece for other models
    if path.is_file():
        return SentencePieceWrapper(tokenizer_model_path)
    else:
        raise ValueError(f"For SentencePiece models, tokenizer_model_path must be a file, got: {tokenizer_model_path}")
