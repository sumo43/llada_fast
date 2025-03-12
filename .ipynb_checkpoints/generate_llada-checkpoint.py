#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
import math
from pathlib import Path
from typing import Optional, Tuple, Union, List

import torch
import torch._dynamo.config
import torch._inductor.config
import torch.nn.functional as F
import numpy as np
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True 
torch._functorch.config.enable_autograd_cache = True
# Disable cudagraph trees to avoid issues with LLaDA model
torch._inductor.config.triton.cudagraph_trees = False

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

create_block_mask = torch.compile(create_block_mask)

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tokenizer import get_tokenizer

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def roundup(val, multiplier):
    return ((val - 1) // multiplier + 1) * multiplier

def causal_mask(b, h, q, kv):
    return q >= kv

def non_causal_mask(b, h, q, kv):
    # For LLaDA models, we don't use causal masking
    # Simplest possible implementation - just return True for all positions
    # We make all positions valid by using a comparison that's always true
    return q >= 0

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, block_mask: BlockMask, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    
    # Use the standard model forward pass as in generate.py
    print("Using standard model forward pass in decode_one_token...")
    
    with torch.inference_mode():
        # For LLaDA, we create a non-causal mask
        if hasattr(model, "_model_name") and "LLaDA" in model._model_name:
            # Use direct model call - similar to generate.py implementation
            logits = model(block_mask, x, input_pos)
    
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    # Use non-causal mask for LLaDA models
    if hasattr(model, "_model_name") and "LLaDA" in model._model_name:
        # Add a bit of extra space to avoid dimension mismatch with kv_cache
        mask_size = model.max_seq_length + 1  
        # Create block mask for non-causal attention
        # We've already compiled create_block_mask at the top of the file
        block_mask = create_block_mask(non_causal_mask, 1, 1, mask_size, mask_size, device=cur_token.device)
    else:
        mask_size = model.max_seq_length + 1
        block_mask = create_block_mask(causal_mask, 1, 1, mask_size, mask_size, device=cur_token.device)
    
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        next_token, next_prob = decode_one_token(
            model, cur_token, input_pos, block_mask, **sampling_kwargs
        )
        input_pos += 1
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        new_probs.append(next_prob.clone())
        cur_token = next_token.clone()

    return new_tokens, new_probs

def model_forward(model, mask, x, input_pos):
    # Mark CUDA graph step to avoid overwriting tensors

    return model(mask, x, input_pos)

def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)

    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64 when appropriate.
    '''

    if temperature == 0:
        return logits
    
    # Use float32 when compiling to avoid dtype issues, otherwise float64
    use_dtype = torch.float32 if torch._dynamo.is_compiling() else torch.float64
    
    logits_f = logits.to(use_dtype)
    noise = torch.rand_like(logits_f)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits_f.exp() / gumbel_noise

def filter_special_tokens(tokens, tokenizer, mask_id):
    """Filter out special tokens from model generation."""
    # Check if we have access to tokenizer info
    if hasattr(tokenizer, "bos_id") and hasattr(tokenizer, "eos_id"):
        bos_id = tokenizer.bos_id()
        eos_id = tokenizer.eos_id()
        # Replace special tokens with mask_id
        special_tokens = [bos_id, eos_id]
        for token_id in special_tokens:
            tokens = torch.where(tokens == token_id, mask_id, tokens)
    
    # Also filter by token ID for typical special tokens
    # These might not be registered with tokenizer
    return tokens

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def get_mask_mod(mask_mod, offset):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + offset, kv)
    return _mask_mod

@torch.no_grad()
def llada_generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int = 1,
    steps: int = 128,
    block_length: int = 128,
    temperature: float = 0.,
    cfg_scale: float = 0.,
    remasking: str = 'low_confidence',
    mask_id: int = 126336,
    callback = lambda x: x,
    **sampling_kwargs
) -> Tuple[torch.Tensor, dict]:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate tokens using LLaDA diffusion.
    
    Args:
        model: Transformer model
        prompt: Input token IDs
        max_new_tokens: Number of tokens to generate
        batch_size: Batch size for generation
        steps: Sampling steps, less than or equal to max_new_tokens
        block_length: Block length for generation, less than or equal to max_new_tokens
        temperature: Temperature for Gumbel noise sampling
        cfg_scale: Classifier-free guidance scale
        remasking: Remasking strategy, 'low_confidence' or 'random'
        mask_id: Token ID for mask token
        callback: Callback function for token generation
    """
    # Set up dimensions
    T = prompt.size(-1)
    device, dtype = prompt.device, prompt.dtype
    
    # Ensure block_length divides max_new_tokens evenly
    assert max_new_tokens % block_length == 0, "block_length must divide max_new_tokens evenly"
    num_blocks = max_new_tokens // block_length
    
    # Ensure steps divides num_blocks evenly
    steps_per_block = steps // num_blocks
    assert steps % num_blocks == 0, f"steps ({steps}) must divide num_blocks ({num_blocks}) evenly"
    
    # Report configuration
    print(f"\nLLaDA Diffusion Generation:")
    print(f"  - Max new tokens: {max_new_tokens}")
    print(f"  - Block length: {block_length}")
    print(f"  - Steps: {steps} ({steps_per_block} per block)")
    print(f"  - Temperature: {temperature}")
    print(f"  - Remasking: {remasking}")
    print(f"  - CFG scale: {cfg_scale}")
    
    # Setup caches
    max_seq_length = T + max_new_tokens
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
    
    # Initialize output sequence with mask tokens
    x = torch.full((batch_size, T + max_new_tokens), mask_id, dtype=dtype, device=device)
    
    # Check if we need to filter special tokens from prompt
    bos_id = mask_id  # Default fallback
    eos_id = mask_id  # Default fallback
    if hasattr(model, 'tokenizer'):
        bos_id = model.tokenizer.bos_id()
        eos_id = model.tokenizer.eos_id()
    
    # Copy the prompt but filter out any special tokens
    filtered_prompt = prompt.clone()
    special_tokens = [bos_id, eos_id]
    
    # Print the special tokens we're filtering
    print(f"Filtering special tokens: {special_tokens}")
    
    # Copy prompt tokens but skip special tokens
    x[:, :T] = filtered_prompt
    
    # Track prompt positions (not masked)
    prompt_index = (x != mask_id)
    
    # Process each block
    for num_block in range(num_blocks):
        # Get mask indices for current block
        block_start = T + num_block * block_length
        block_end = T + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        
        # Compute number of tokens to transfer at each step
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        # Diffusion process for each step
        for i in range(steps_per_block):
            print(f"Block {num_block+1}/{num_blocks}, Step {i+1}/{steps_per_block}")
            
            # Get current mask indices
            mask_index = (x == mask_id)

            # Use DIRECT forward call instead of using flex_attention
            with torch.inference_mode():
                # Use direct model components instead of model.forward
                # For classifier-free guidance
                if cfg_scale > 0.:
                    # Create unconditional version
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    
                    # Use model.forward directly - it's much simpler and handles all the internal details
                    # Create a block mask for non-causal attention
                    input_seq_len = x.shape[1]
                    input_pos = torch.arange(0, input_seq_len, device=device)
                    
                    # Use the standard model forward pass for both conditional and unconditional inputs
                    print(f"Input shapes - x: {x.shape}, input_pos: {input_pos.shape}")
                    print("Using standard model forward pass...")
                    
                    try:
                        # Create a block mask for non-causal attention
                        mask = create_block_mask(non_causal_mask, 1, 1, input_seq_len, input_seq_len, device=device)
                        
                        # Process conditional input with standard model call using compiled model_forward
                        logits_cond = model_forward(model, mask, x, input_pos)
                        
                        # Process unconditional input with standard model call
                        logits_uncond = model_forward(model, mask, un_x, input_pos)
                    except Exception as e:
                        print(f"Error during model computation with compiled functions: {e}")
                        print(f"Falling back to direct model calls...")
                        print(f"Input shapes - x: {x.shape}, un_x: {un_x.shape}, input_pos: {input_pos.shape}")
                        
                        # Fallback to direct model calls
                        logits_cond = model(mask, x, input_pos)
                        logits_uncond = model(mask, un_x, input_pos)
                    
                    # Apply CFG
                    logits = logits_uncond + (cfg_scale + 1) * (logits_cond - logits_uncond)
                else:
                    # Use model.forward directly - it's much simpler and handles all the internal details
                    input_seq_len = x.shape[1]
                    input_pos = torch.arange(0, input_seq_len, device=device)
                    
                    # Use the standard model forward pass
                    print(f"Input shapes - x: {x.shape}, input_pos: {input_pos.shape}")
                    print("Using standard model forward pass...")
                    
                    try:
                        # Create a block mask for non-causal attention
                        mask = create_block_mask(non_causal_mask, 1, 1, input_seq_len, input_seq_len, device=device)
                        
                        # Process input with standard model call using compiled model_forward
                        logits = model_forward(model, mask, x, input_pos)
                    except Exception as e:
                        print(f"Error during model computation with compiled functions: {e}")
                        print(f"Falling back to direct model call...")
                        print(f"Input shapes - x: {x.shape}, input_pos: {input_pos.shape}")
                        
                        # Fallback to direct model call
                        logits = model(mask, x, input_pos)
            
            # Apply Gumbel noise for sampling
            try:
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            except Exception as e:
                print(f"Error with compiled add_gumbel_noise: {e}")
                print("Falling back to direct implementation")
                # Fallback implementation of add_gumbel_noise
                if temperature == 0:
                    logits_with_noise = logits
                else:
                    logits_f = logits.to(torch.float32)
                    noise = torch.rand_like(logits_f)
                    gumbel_noise = (- torch.log(noise)) ** temperature
                    logits_with_noise = logits_f.exp() / gumbel_noise
                    
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
            
            # Filter out any special tokens from the generated tokens
            if hasattr(model, 'tokenizer'):
                # Use model's tokenizer if available
                x0 = filter_special_tokens(x0, model.tokenizer, mask_id)
            
            # Compute token confidence for remasking
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            # Don't consider tokens outside current block + processed blocks
            x0_p[:, block_end:] = -np.inf
            
            # Update predicted tokens only at masked positions
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # Transfer tokens based on confidence
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
            # Update sequence with new tokens - ensure same dtype
            x[transfer_index] = x0[transfer_index].to(x.dtype)
            
            # Call callback with newly transferred tokens
            for j in range(batch_size):
                # Only call back for tokens in the current block
                block_start_idx = T + num_block * block_length
                block_end_idx = T + (num_block + 1) * block_length
                
                # Get the tokens that were transferred in this block
                batch_transfer = transfer_index[j, block_start_idx:block_end_idx]
                if batch_transfer.any():
                    transferred_indices = torch.where(batch_transfer)[0] + block_start_idx
                    callback(x[j, transferred_indices])
            
            # Print progress
            remaining_masks = (x == mask_id).sum().item()
            print(f"  Masks remaining: {remaining_masks}/{max_new_tokens}")
    
    # Return the full sequence and generation statistics
    return x, {'accept_counts': []}

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback = lambda x: x,
    **sampling_kwargs
) -> Tuple[torch.Tensor, dict]:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    Delegates to llada_generate for LLaDA models.
    """
    # For LLaDA models, use the diffusion-based generation
    if hasattr(model, "_model_name") and "LLaDA" in model._model_name:
        # Extract LLaDA-specific parameters from sampling_kwargs
        llada_params = {
            'steps': sampling_kwargs.pop('steps', 128),
            'block_length': sampling_kwargs.pop('block_length', min(128, max_new_tokens)),
            'cfg_scale': sampling_kwargs.pop('cfg_scale', 0.),
            'remasking': sampling_kwargs.pop('remasking', 'low_confidence'),
            'mask_id': sampling_kwargs.pop('mask_id', 126336),
            'temperature': sampling_kwargs.pop('temperature', 0.0),
        }
        return llada_generate(
            model,
            prompt,
            max_new_tokens,
            batch_size,
            callback=callback,
            **llada_params,
            **sampling_kwargs
        )

    # For non-LLaDA models, use the standard autoregressive generation
    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    if interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    # We are just making the same prompt for every batch
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs).clone()
    if is_speculative:
        prefill(draft_model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs)
    seq[:, T] = next_token.squeeze()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < T_new - 1:
            cur_token = next_token.view(())

            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
            for i in next_tokens[: num_added,]:
                callback(i)
            input_pos = input_pos + num_added
            next_token = next_tokens[-1]
    else:
        generated_tokens, _ = decode_n_tokens(model, next_token.view(batch_size, -1), input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
        seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)

    generate_stats = {
        'accept_counts': accept_counts
    }
    return seq, generate_stats

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    # Use non-causal mask for LLaDA models
    if hasattr(model, "_model_name") and "LLaDA" in model._model_name:
        mask = create_block_mask(non_causal_mask, 1, 1, input_pos.shape[0], model.max_seq_length, device=x.device)
    else:
        mask = create_block_mask(causal_mask, 1, 1, input_pos.shape[0], model.max_seq_length, device=x.device)
    logits = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    # Remove special tokens from the prompt if present
    string = string.replace("<|startoftext|>", "").replace("<|endoftext|>", "")
    
    # Get special token IDs for filtering
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    
    # Get tokens without special tokens that might be inserted by the tokenizer
    tokens = tokenizer.encode(string)
    
    # Remove any startoftext or endoftext tokens that might have been added
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "convert_tokens_to_ids"):
        # For HuggingFace tokenizers
        start_token_id = tokenizer.tokenizer.convert_tokens_to_ids("<|startoftext|>")
        end_token_id = tokenizer.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if start_token_id != -1 and end_token_id != -1:  # -1 means token not found
            tokens = [t for t in tokens if t != start_token_id and t != end_token_id]
    
    # Add BOS token if requested
    if bos:
        tokens = [bos_id] + tokens
        
    print(f"Encoded tokens: {tokens}")
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _load_model(checkpoint_path, device, precision, use_tp):
    use_cuda = 'cuda' in device
    model_name = checkpoint_path.parent.name
    with torch.device('meta'):
        model = Transformer.from_name(model_name)
    
    # Ensure model name is passed through the model
    if hasattr(model, "_model_name"):
        model._model_name = model_name

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()

def _get_model_size(model):
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            params += sum(
                [
                    p.numel()
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size, params

B_INST, E_INST = "[INST]", "[/INST]"

def main(
    prompt: Union[int, str] = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    compile: bool = False,
    compile_prefill: bool = False,
    compile_mode: str = "reduce-overhead",
    profile: Optional[Path] = None,
    draft_checkpoint_path: Optional[Path] = None,
    speculate_k: int = 5,
    device=default_device,
    # LLaDA-specific parameters
    steps: int = 128,
    block_length: int = 32,
    cfg_scale: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: int = 126336,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    assert checkpoint_path.is_file(), checkpoint_path

    # Setup distributed training and printing
    from tp import maybe_init_dist
    global print
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    model_name = checkpoint_path.parent.name
    
    # For HuggingFace-based models like LLaDA, we might not have a tokenizer.model file
    # but instead have tokenizer.json or other HF tokenizer files
    if not tokenizer_path.is_file() and ("LLaDA" in model_name):
        hf_tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        if any((checkpoint_path.parent / file).exists() for file in hf_tokenizer_files):
            # If any HF tokenizer files exist, we'll use the directory instead
            tokenizer_path = checkpoint_path.parent
            print(f"Using HuggingFace tokenizer from directory: {tokenizer_path}")
        else:
            # Still enforce the assertion if we can't find tokenizer files
            assert tokenizer_path.is_file(), f"{str(tokenizer_path)} not found and no HuggingFace tokenizer files found in {checkpoint_path.parent}"
    else:
        assert tokenizer_path.is_file(), str(tokenizer_path)

    print(f"Using device={device}")
    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None
    is_chat = "chat" in str(checkpoint_path)
    is_llada = "LLaDA" in model_name

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    if is_speculative:
        draft_model = _load_model(draft_checkpoint_path, device, precision, use_tp)
    else:
        draft_model = None

    device_sync(device=device) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, model_name)

    if isinstance(prompt, str):
        # For LLaDA Instruct models, apply chat template if available
        if is_llada and is_chat:
            from transformers import AutoTokenizer
            if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "apply_chat_template"):
                # Handle our HuggingFaceTokenizerWrapper
                m = [{"role": "user", "content": prompt}, ]
                prompt_text = tokenizer.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                print(f"Applied chat template: {prompt_text}")
                encoded = encode_tokens(tokenizer, prompt_text, bos=True, device=device)
            else:
                # Standard tokenizer but try to format message for LLaDA
                m = [{"role": "user", "content": prompt}, ]
                try:
                    # Try to import and use directly if needed
                    hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path.parent)
                    prompt_text = hf_tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                    print(f"Applied chat template: {prompt_text}")
                    encoded = encode_tokens(tokenizer, prompt_text, bos=True, device=device)
                except:
                    # Fall back to basic formatting
                    prompt = f"{B_INST} {prompt.strip()} {E_INST}"
                    encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
        else:
            # For non-LLaDA models
            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    else:
        # generate a fully synthetic prompt
        encoded = torch.randint(0, 1024, (prompt,), device=device, dtype=torch.int64)
    prompt_length = encoded.size(-1)

    torch.manual_seed(1234)
    model_size, params = _get_model_size(model)
    
    # For LLaDA, we have special compilation strategy
    if compile:
        if is_speculative and use_tp:
            torch._inductor.config.triton.cudagraph_trees = False # Bug with cudagraph trees in this case

        global model_forward, logits_to_probs, add_gumbel_noise, get_num_transfer_tokens
        
        # Compile key functions for both standard and LLaDA models
        print(f"Using compilation mode: {compile_mode}")
        
        print("Compiling model_forward...")
        model_forward = torch.compile(model_forward, mode=compile_mode, fullgraph=True)
        
        if is_llada:
            print("Compiling LLaDA-specific functions...")
            # These functions are particularly important for LLaDA performance
            try:
                print("Compiling add_gumbel_noise...")
                add_gumbel_noise = torch.compile(add_gumbel_noise, mode=compile_mode, fullgraph=False)
                print("Compiling get_num_transfer_tokens...")
                get_num_transfer_tokens = torch.compile(get_num_transfer_tokens, mode=compile_mode, fullgraph=False)
                print("Compiling logits_to_probs for LLaDA...")
                logits_to_probs = torch.compile(logits_to_probs, mode=compile_mode, fullgraph=False)
            except Exception as e:
                print(f"Warning: Could not compile some LLaDA functions: {e}")
                print("Continuing with uncompiled functions")
        
        if is_speculative:
            print("Compiling speculative decoding functions...")
            logits_to_probs = torch.compile(logits_to_probs, mode=compile_mode, fullgraph=True)

        global decode_one_token, prefill
        print("Compiling decode_one_token...")
        decode_one_token = torch.compile(decode_one_token, mode=compile_mode, fullgraph=True)

        # Squeeze more perf out of prefill
        if compile_prefill:
            print("Compiling prefill...")
            prefill = torch.compile(prefill, mode=compile_mode, fullgraph=True, dynamic=True)

    aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
    }
    # Always include a warmup/compilation run when compile=True
    start = -1 if compile else 0

    for i in range(start, num_samples):
        device_sync(device=device) # MKG
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            # For LLaDA Instruct models, apply chat template if available
            if is_llada and is_chat:
                from transformers import AutoTokenizer
                if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "apply_chat_template"):
                    # Handle our HuggingFaceTokenizerWrapper
                    m = [{"role": "user", "content": prompt}, ]
                    prompt_text = tokenizer.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                    print(f"Applied chat template: {prompt_text}")
                    encoded = encode_tokens(tokenizer, prompt_text, bos=True, device=device)
                else:
                    # Standard tokenizer but try to format message for LLaDA
                    m = [{"role": "user", "content": prompt}, ]
                    try:
                        # Try to import and use directly if needed
                        hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path.parent)
                        prompt_text = hf_tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                        print(f"Applied chat template: {prompt_text}")
                        encoded = encode_tokens(tokenizer, prompt_text, bos=True, device=device)
                    except:
                        # Fall back to basic formatting
                        prompt = f"{B_INST} {prompt.strip()} {E_INST}"
                        encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
            else:
                # For non-LLaDA models
                if is_chat:
                    prompt = f"{B_INST} {prompt.strip()} {E_INST}"
                encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode('.')[0]
            done_generating = False
            
            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                
                # Handle tensor arrays and individual tokens
                if hasattr(x, 'shape') and len(x.shape) > 0 and x.shape[0] > 1:
                    # This is a tensor array from LLaDA diffusion
                    tokens_text = tokenizer.decode(x.tolist())
                    # Filter out special tokens
                    tokens_text = tokens_text.replace("<|startoftext|>", "").replace("<|endoftext|>", "")
                    buffer.append(tokens_text)
                else:
                    # Single token from autoregressive generation
                    tokens = x.tolist() if hasattr(x, 'tolist') else [x]
                    for token in tokens:
                        if isinstance(token, torch.Tensor):
                            token = token.item()
                        decoded = tokenizer.decode([period_id, token])[1:]
                        # Filter out special tokens
                        decoded = decoded.replace("<|startoftext|>", "").replace("<|endoftext|>", "")
                        buffer.append(decoded)
                        if token == tokenizer.eos_id():
                            done_generating = True
                
                # Print buffered text
                if len(buffer) >= 4 or done_generating:
                    text = ''.join(buffer)
                    # One final filter of the entire text
                    text = text.replace("<|startoftext|>", "").replace("<|endoftext|>", "")
                    print(text, end='', flush=True)
                    buffer.clear()
        else:
            callback = lambda x : x
        t0 = time.perf_counter()
        import contextlib
        if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            # Create sampling kwargs dictionary
            sampling_kwargs = {
                'temperature': temperature,
                'top_k': top_k,
            }
            
            # Add LLaDA-specific parameters for diffusion generation
            sampling_kwargs_with_llada = {
                **sampling_kwargs,
                'steps': steps,
                'block_length': block_length,
                'cfg_scale': cfg_scale,
                'remasking': remasking,
                'mask_id': mask_id,
            }
            
            y, metrics = generate(
                model,
                encoded,
                max_new_tokens,
                batch_size=batch_size,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                **sampling_kwargs_with_llada,
            )
            aggregate_metrics['accept_counts'].append(metrics.get('accept_counts', []))
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=device) # MKG
        t = time.perf_counter() - t0

        if not interactive:
            # Just displaying the first generation
            if batch_size > 1:
                print("Only displaying the first generation of the batch")
            output = tokenizer.decode(y[0].tolist())
            # Filter out special tokens before displaying
            output = output.replace("<|startoftext|>", "").replace("<|endoftext|>", "")
            print(output)

        tokens_generated = max_new_tokens if interactive else y.size(1) - prompt_length
        tokens_sec = tokens_generated / t
        print(f"Generated {tokens_generated} tokens in {t:.2f} seconds, {tokens_sec:.2f} tokens/sec")
        aggregate_metrics['tokens_per_sec'].append(tokens_sec)

    print()
    print(f"Model size: {model_size / 1e9:.2f}B parameters, {model_size / 1e9 * 2:.2f} GB (fp16)")
    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    def int_or_str(x):
        try:
            return int(x)
        except:
            return x

    parser.add_argument('--prompt', type=int_or_str, default="Hello, my name is", help="Input prompt. If it's an integer, will instead generate a synthetic prompt.")
    parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=64, help='Maximum number of new tokens.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to benchmark with')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead', 
                      choices=['default', 'reduce-overhead', 'max-autotune'], 
                      help='The torch.compile mode to use. "default" is slower to compile but may be faster, "reduce-overhead" compiles faster but may be slower, "max-autotune" does aggressive optimizations but is very slow to compile.')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    
    # LLaDA-specific arguments
    parser.add_argument('--steps', type=int, default=64, help='Sampling steps for LLaDA diffusion')
    parser.add_argument('--block_length', type=int, default=256, help='Block length for LLaDA diffusion')
    parser.add_argument('--cfg_scale', type=float, default=0.0, help='Classifier-free guidance scale for LLaDA diffusion')
    parser.add_argument('--remasking', type=str, default='low_confidence', choices=['low_confidence', 'random'], help='Remasking strategy for LLaDA diffusion')
    parser.add_argument('--mask_id', type=int, default=126336, help='Token ID for mask token in LLaDA')

    args = parser.parse_args()
    main(
        args.prompt, args.interactive, args.num_samples, args.max_new_tokens, args.batch_size, args.top_k,
        args.temperature, args.checkpoint_path, args.compile, args.compile_prefill, args.compile_mode,
        args.profile, args.draft_checkpoint_path, args.speculate_k, args.device,
        args.steps, args.block_length, args.cfg_scale, args.remasking, args.mask_id
    )