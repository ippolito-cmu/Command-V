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

# LIMA
## Chinese_LIMA_Qs
Chinese_LIMA_Qs.txt contains 1030 lines of translated LIMA prompts from training.json.
It only translated the first element in the conversation of the original LIMA dataset.

## Access
Since LIMA is gated, you must obtain a LIMA access first.
You can obtain access to LIMA data on https://huggingface.co/datasets/GAIR/lima
Then download https://huggingface.co/datasets/GAIR/lima/raw/main/train.jsonl and get its md5 value
Which should be `***************704059`. This is the password for the zip.

## Translation Details
Translation are done using Claude 3.5 Sonnet in September 2024, since it seems to provide the most natural results.
Here is the prompt preceding the LIMA question (one per line) lines in batch of 25-100:
```
Translate each line to Chinese without losing any details, but use native Chinese expressions wherever possible. Make sure it sounds like a question that Chinese people would actually say/ask in real life and on 知乎 or 豆瓣 or 百度知道 etc. rather than a verbose word-by-word translated sentence. preserve all of the \n and other symbols in the translation. Output as an artifcat. One line per original line.
```
Postprocessing including adjusting any translation request to Chinese, replacing [,!?] with [，！？] if following a Chinese character.