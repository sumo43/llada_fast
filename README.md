# gpt-fast
Simple and efficient LLaDA implementation

## Downloading Weights
Models tested/supported
```text
GSAI-ML/LLaDA-8B-Instruct
GSAI-ML/LLaDA-8B-Base
```

For example, to convert Llama-2-7b-chat-hf
```bash
export MODEL_REPO=GSAI-ML/LLaDA-8B-Instruct
./scripts/prepare.sh $MODEL_REPO
```

## Benchmarks

When generating 512 tokens on a single H100:
```
- 169 tok/s with 128 sampling steps, block_length 32
- 344 tok/s with 64 sampling steps, block_length 32
- 696 tok/s with 32 sampling steps, block_length 32
```


## Generate Text

Model definition in `model.py`, generation code in `generate_llada.py`.

```bash
python generate_llada.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"
```

To squeeze out a little bit more performance, you can also compile the prefill with `--compile_prefill`. This will increase compilation times though.


## Tensor Parallelism
```bash
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=2 generate_llada.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth
```

## License

`llada-fast` is released under the [BSD 3](https://github.com/pytorch-labs/gpt-fast/main/LICENSE) license.

## Acknowledgements
Thanks to:
* gpt_fast codebase for the fast, simple llama implementation
