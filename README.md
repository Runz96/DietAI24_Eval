# DietAI24 — Quick Start

**Dataset**  
- **Nutribench**: text-only food descriptions.  
- **Kenya**: food images.  
Each `kenya/*.py` or `nutribench/*.py` is a model adapter. **Main runners**: `DietAI24_kenya.ipynb`, `DietAI24_nutribench.ipynb`.

---

## Supported Models
**Nutribench (LLMs)**: GPT‑4.1, Claude Sonnet 4, DeepSeek‑R1, Llama‑3.1‑8B.  
**Kenya (MLLMs)**: GPT‑4.1, Claude Sonnet 4, DeepSeek‑VL2 *(tested: `deepseek-vl2-tiny`)*, LLaVA‑Next *(tested: `llava-v1.6-mistral-7b-hf`)*.  
Model cards:  
- DeepSeek‑VL2‑Tiny → https://huggingface.co/deepseek-ai/deepseek-vl2-tiny  
- LLaVA‑Next v1.6 (Mistral‑7B) → https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf

---

---

## Data
- **Kenya**: images under `data/kenya/kenya_parsed_food_images/`.  
- **Nutribench**: text descriptions under `data/nutribench/selected_nutribench_v2.csv`.

---

## Run (notebooks)
Open **`DietAI24_kenya.ipynb`** or **`DietAI24_nutribench.ipynb`** and choose ONE adapter.

**Kenya (example)**
```python
from kenya.DietAI24_OpenAI import DietAI24  
recognizer = DietAI24(model_name=model_name, food_list_path=path_food_list)
```

**Nutribench (example)**
```python
from nutribench.DietAI24_OpenAI import DietAI24   # or: DietAI24_Anthropic / DietAI24_DeepSeek / DietAI24_Llama
recognizer = DietAI24(model_name=model_name, vectordb=vectordb)
```

**Switching models** = swap the import line, nothing else.

---

## Outputs
CSV in `outputs/` (predicted foods/codes, grams, and model metadata).

---

**Notes**  
- First run of local HF models will download weights; GPU recommended.  
- Keep runs reproducible by logging model IDs and seeds.