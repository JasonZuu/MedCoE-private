## LLM EHR Input Exploration
This project aims to explore the most effective way to input EHR data into the LLM. Specifically, we aims to explore the following things
1. The stucture of EHR data
   + Regular time-series
   + Irregular time-series
   + Event Stream

2. The format of the structured EHR data
   + NL + <sep>
   + Json
   + XML
   + HTML

3. The Usage of LLM
   + Predictor
   + Embedding Model


## Data Processing
This project trasform the EHR data to event stream using the (MEDS)[https://github.com/Medical-Event-Data-Standard/]. Specifically, we processed the data with the following steps:
### 1. MEDS-Transform: turn the EHR into event stream

### 2. ACES-based task cohort extraction.
Please refer to the *extract_cohort.py* in [MEDS_extract](https://github.com/JasonZuu/MEDS_extract)
### 3. Turn the data into LLM compatible QA format
Please refer to *construct_qa.py* in [MEDS_extract](https://github.com/JasonZuu/MEDS_extract)
```
pass
```

## Download Models
1. Download from huggingface hub using *huggingface-cli*
```
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ~/Data/hf_models/Qwen--Qwen2.5-7B-Instruct 
huggingface-cli download OpenMeditron/Meditron3-Qwen2.5-7B --local-dir ~/Data/hf_models/OpenMeditron--Meditron3-Qwen2.5-7B
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir ~/Data/hf_models/deepseek-ai--DeepSeek-R1-Distill-Qwen-7B

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ~/Data/hf_models/meta-llama--Llama-3.1-8B-Instruct
huggingface-cli download OpenMeditron/Meditron3-8B --local-dir ~/Data/hf_models/OpenMeditron--Meditron3-8B
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B --local-dir ~/Data/hf_models/deepseek-ai--DeepSeek-R1-Distill-Llama-8B
```