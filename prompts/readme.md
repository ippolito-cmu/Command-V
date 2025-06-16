# AdvBench
test.txt is downloaded from
https://github.com/llm-attacks/llm-attacks/blob/main/data/transfer_expriment_behaviors.csv

Accessed April 2025

MIT Licensed


# AGI-Edgerunners--LLM-Adapters-dataset
Downloaded from https://github.com/AGI-Edgerunners/LLM-Adapters

Accessed March 2025

Apache-2.0 licensed

# allenai--wildjailbreak
Downloaded from Huggingface 

Accessed March 2025

ODC-By licensed. Additional Agreement Required

Mandatory agreement: https://allenai.org/responsible-use

Processing includes flattening json dict to list.

# Big Bench Hard
Downloaded from Huggingface (maveriq/bigbenchhard)

Accessed April 2025

MIT licensed

# facebook--natural_reasoning
Downloaded from Huggingface

Accessed April 2025

CC-BY-NC-4.0

## Processing applied
We find questions and answers where the correct answer is non-empty and appears verbatim in the first sample model response. 
Sort the dataset by prompt+answer token count (BERT tokenizer) and keep only those have than 1000 tokens or fewer. 
Use numpy shuffle with seed 42. Use the first 90% as train.json (42366 pairs) and 10% as test.json (4708 pairs).

# Harm Bench
Downloaded from https://github.com/centerforaisafety/HarmBench

Accessed March 2025

MIT licensed
