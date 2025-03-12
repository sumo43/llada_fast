# gpt-fast
Simple and efficient LLaDA

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



## Generate Text

Model definition in `model.py`, generation code in `generate.py`.

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
