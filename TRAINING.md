# Guidelines
Here, we provide guidelines for the model architecture, pre-training, SFT, and inference of LLaDA.

## Model Architecture

LLaDA employs a Transformer Encoder as the network architecture for its mask predictor. 
In terms of trainable parameters, the Transformer Encoder is identical to the Transformer 
Decoder. Starting from an autoregressive model, we derive the backbone of LLaDA by simply 
removing the causal mask from the self-attention mechanism as following.

<div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 50px;">
    <img src="imgs/transformer1.png" style="width: 90%;" />
    <img src="imgs/transformer2.png" style="width: 90%;" />
</div>

In addition, LLaDA designates a reserved token as the mask token (i.e., 126336).


## Pre-training
The pre-training of LLaDA is straightforward and simple. Starting from an existing 
autoregressive model training code, only a few lines need to be modified. 
We provide the core code (i.e., loss computation) here.

```angular2html
def forward_process(input_ids, eps=1e-3):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    # 126336 is used for [MASK] token
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    return noisy_batch, masked_indices, p_mask

# The data is an integer tensor of shape (b, 4096), 
# where b represents the batch size and 4096 is the sequence length.
input_ids = batch["input_ids"]

# We set 1% of the pre-training data to a random length that is uniformly sampled from the range [1, 4096].
# The following implementation is not elegant and involves some data waste. 
# However, the data waste is minimal, so we ignore it.
if torch.rand(1) < 0.01:
    random_length = torch.randint(1, input_ids.shape[1] + 1, (1,))
    input_ids = input_ids[:, :random_length]

noisy_batch, masked_indices, p_mask = forward_process(input_ids)
logits = model(input_ids=noisy_batch).logits

token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

```

## SFT
First, please refer to Appendix B.1 for the preprocessing of the SFT data. After preprocessing the data, 
the data format is as follows. For simplicity, we treat each word as a token and set the batch size to 2 
in the following visualization.
```angular2html
input_ids:
<BOS><start_id>user<end_id>\nWhat is the capital of France?<eot_id><start_id>assistant<end_id>\nParis.<EOS><EOS><EOS><EOS><EOS><EOS><EOS><EOS><EOS><EOS>
<BOS><start_id>user<end_id>\nWhat is the capital of Canada?<eot_id><start_id>assistant<end_id>\nThe capital of Canada is Ottawa, located in Ontario.<EOS>

prompt_lengths:
[17, 17]
```

After preprocessing the SFT data, we can obtain the SFT code by making simple modifications to the pre-training code. 
The key difference from pre-training is that SFT does not add noise to the prompt.
```angular2html
input_ids, prompt_lengths = batch["input_ids"], batch["prompt_lengths"]

noisy_batch, _, p_mask = forward_process(input_ids)

# Do not add noise to the prompt
token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
prompt_mask = (temp_tensor < prompt_length.unsqueeze(1))
noisy_batch[prompt_mask] = input_ids[prompt_mask]

# Calculate the answer length (including the padded <EOS> tokens)
prompt_mask = prompt_mask.to(torch.int64)    
answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
answer_lengths = answer_length.repeat(1, noisy_batch.shape[1])    

masked_indices = (noisy_batch == 126336)

logits = model(input_ids=noisy_batch).logits
    
token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
```

## Sampling
Overall, we categorize LLaDA's sampling process into three types: fixed-length, semi-autoregressive-origin, and semi-autoregressive-padding.
**It is worth noting that the semi-autoregressive-origin method was not mentioned in our paper, nor did we provide the corresponding code**. 
However, we include it here because we believe that sharing both our failures and insights from the exploration process is valuable.
These three sampling methods are illustrated in the figure below.


<div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 50px;">
    <img src="imgs/sample.png" style="width: 100%;" />
</div>

For each step in the above three sampling processes, as detailed in Section 2.4 in our paper, the mask predictor 
first predicts all masked tokens simultaneously. Then, a certain proportion of these predictions are remasked. 
To determine which predicted tokens should be re-masked, we can adopt two strategies: *randomly remasking* or 
*low-confidence remasking*. Notably, both remasking strategies can be applied to all three sampling processes 
mentioned above.

For the LLaDA-Base model, we adapt low-confidence remasking to the three sampling processes mentioned above. 
We find that fixed-length and semi-autoregressive-padding achieve similar results, whereas semi-autoregressive-origin 
performs slightly worse.

For the LLaDA-Instruct model, the situation is slightly more complex. 

First, if the semi-autoregressive-origin method is used, 
the Instruct model performs poorly. This is because, during SFT, each sequence is a complete sentence (whereas in pre-training, 
many sequences are truncated sentences). As a result, during sampling, given a generated length, regardless of whether it is 
long or short, the Instruct model tends to generate a complete sentence. Unlike the Base model, it does not encounter cases
where a sentence is only partially generated and needs to be continued.

When performing fixed-length sampling with a high answer length (e.g., greater than 512), 
we find that low-confidence remasking results in an unusually high proportion of `<EOS>` tokens in 
the generated sentences, which severely impacts the model's performance. In contrast, this 
issue does not arise when randomly remasking is used.

Furthermore, since low-confidence remasking achieved better results in the Base model, we also hoped that it could be applied to 
the Instruct model. We found that combining low-confidence remasking with semi-autoregressive-padding effectively mitigates 
the issue of generating an excessively high proportion of <EOS> tokens. Moreover, this combination achieves 
slightly better results than randomly remasking & fixed-length.

You can find more details about the sampling method in our paper.







