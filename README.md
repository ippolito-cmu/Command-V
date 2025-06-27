# Command-V: Pasting LLM Behaviors via Activation Profiles 
[![arXiv](https://img.shields.io/badge/arXiv-Preprint%20available-red.svg)](https://arxiv.org/abs/2506.19140)

**Finetune once, use on many LLMs.** ⌘V is the first to demonstrate that you can "copy-paste" a finetuned adapter (for refusal suppression, jailbreaking resistance, chain-of-thought reasoning, etc.) learned by one LLM to another of a different size or architecture, without backpropagation.

![PDF Page 1](commandv/Overview.png)
### How it works
⌘V leverages the fact that all transformers LLM activations (intermediate representations) can be intervened for behaviors using PEFT and approximated from another LLM. So to get it running:
1. **Profile activations** on both models using a small set of prompts
2. **Create converters** that map between model representation spaces
3. **Transfer behaviors** by applying the donor model's learned interventions to the recipient

See details in our [paper](https://www.arxiv.org/abs/2506.19140).

## ⌘V Highlights

- **Low Resource**: No backpropagation. No expert data. Low GPU memory usage. 
- **Fast**: Minutes to set up, not hours.
- **Cross-Family**: In certain conditions, works between Llama, Qwen, and other model families


## 🪧 Demo

Check out the complete pipeline ([./command_v_demo.ipynb](https://github.com/GithuBarry/Command-V/blob/main/command_v_demo.ipynb)) or on Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GithuBarry/Command-V/blob/main/command_v_demo.ipynb) 


## 🎯 Quick Start

### Installation

```bash
git clone https://github.com/GithuBarry/Command-V.git
cd Command-V
pip install -r requirements.txt
```

### Three-Step Pipeline
```bash
# Profile models on LIMA dataset (once per model)
python step1_capture_activations.py --models meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.1-8B-Instruct
```
```bash
# Learn activation space mappings (takes seconds)
# (Mappings are created on the fly in practice, making this step optional.)
python step2_derive_converters.py derive \
  --source-model meta-llama/Llama-3.1-8B-Instruct \
  --target-model meta-llama/Llama-3.2-1B-Instruct
```
```bash
# Transfer behaviors during inference
python step3_commandv_inference.py \
  --recipient-model meta-llama/Llama-3.2-1B-Instruct \
  --donor-model meta-llama/Llama-3.1-8B-Instruct \
  --adapter-folder reft-adapters/jailbreak/Llama-3.1-8B-Instruct/NodireftIntervention/l1/walledai--AdvBench/L0;2;4;6;8;10;12;14;16;18;20;22;24;26;28;30 \
  --input-source prompts/AdvBench/test.txt \
  --first-n 5 --print-output-only
```

## 📁 Project Structure

```
├── command_v_demo.ipynb         # Jupyter Demo
├── step1_capture_activations.py # Step 1: Activation profiling
├── step2_derive_converters.py   # Step 2: Converter derivation  
├── step3_commandv_inference.py  # Step 3: Behavioral transfer
├── commandv/                    # Core library
│   ├── core/                    # Main functionality
│   │   ├── capture.py           # Activation capture
│   │   ├── converters.py        # Pseudoinverse converters
│   │   └── inference.py         # Inference engine
│   ├── utils/                   # Utilities
│   └── data/                    # Data processing
├── outputs/                     # Generated files
│   ├── activations/             # Model activation profiles
│   ├── converters/              # Converter mappings (not by default)
│   └── inferences/              # Results
└── reft-adapters/               # Trained behavior adapters
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Certain artifacts are further gated for their potential to be abused.
