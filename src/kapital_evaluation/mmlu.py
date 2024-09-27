from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
    
    
def compare_answers(actual_answer: str, predicted_answer: str) -> int:
    return actual_answer.strip().lower() == predicted_answer.strip().lower()
    

def evaluate(model, dtype, tasks, num_fewshot, batch_size, device, limit, write_out):
    
    data = load_dataset("LLM-Beetle/banking_support_azerbaijan_mc")
    
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir='cache')
    model = AutoModelForCausalLM.from_pretrained(model, cache_dir='cache').to("cpu")
    
    generation_config = model.generation_config
    generation_config.max_length = 1000
    generation_config.pad_token_id = tokenizer.pad_token_id
    
    score = 0
    
    if limit == None:
        limit = len(data['train']['text'])
    
    for i in tqdm(range(limit), desc='mmlu'):
        
        question = data['train']['text'][i]
        options = data['train']['options'][i]
        correct_answer = data['train']['answer'][i]
        
        conversation = [
            {
                "role": "system",
                "content": "You are given a statement along with multiple options that represent different topics. Choose the option that best categorizes the statement based on its topic. Select the single option (e.g., A, B, C, etc.) that most accurately describes the topic of the statement."
            },
            {
                "role": "user",
                "content": "Mənə yaxın filial tapmağa kömək edin Options: A) ATM, B) FEES, C) OTHER, D) CARD, E) ACCOUNT, F) TRANSFER, G) PASSWORD, H) LOAN, I) CONTACT, J) FIND"
            },
            {
                "role": "assistant",
                "content": "J"
            },
            {
                "role": "user",
                "content": f"{question} Options: {options}"
            }
        ]

        text_input = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False)

        encoding = tokenizer(text_input, return_tensors="pt").to("cpu")

        with torch.inference_mode():
            outputs = model.generate(
                input_ids = encoding.input_ids,
                attention_mask = encoding.attention_mask,
                generation_config = generation_config,
            )
            
        predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
        predicted_answer = predicted_answer.split('<|end_header_id|>')[-1]
        predicted_answer = predicted_answer.replace('<|eot_id|>', '')

        if compare_answers(correct_answer, predicted_answer):
            score +=1
            
    result = {
        "config": {},
        "results": {
            "MMLU": {
                "metric_name": score/limit
            },
            "Truthful_qa": {
                "metric_name": 0.0
            },
            "ARC": {
                "metric_name": 0.0
            },
            "HellaSwag": {
                "metric_name": 0.0
            },
            "GSM8K": {
                "metric_name": 0.0
            },
            "Winogrande": {
                "metric_name": 0.0
            }
        }
    }
            
    return result

    
if __name__=='__main__':
    
    print(evaluate('LLM-Beetle/ProdataAI_Llama-3.1-8bit-Instruct',0,0,0,0,0,10,0))