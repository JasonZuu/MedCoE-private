from llmlingua import PromptCompressor


llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", # the large model of LLMLingua2
    use_llmlingua2=True, # Whether to use llmlingua-2
)
prompts = ["What is the capital of France? Please provide a detailed answer."] * 10
compressed_prompts = llm_lingua.compress_prompt(prompts, 
                                               rate=1-0.33, 
                                               force_tokens = ['\n', '?'])
print(compressed_prompts["compressed_prompt_list"])
print(type(compressed_prompts["compressed_prompt_list"]))
