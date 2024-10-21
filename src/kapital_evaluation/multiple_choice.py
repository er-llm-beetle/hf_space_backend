from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import re
import pandas as pd

from utility import format_options



def get_answer_multiple_choice(question, options, model, tokenizer, num_fewshot, base_prompt, dstype):

    # generation_config = model.generation_config
    # generation_config.max_length = 1000
    # generation_config.pad_token_id = tokenizer.pad_token_id


    options = format_options(options, dstype) # format option


    conversation = []

    conversation.append(
        base_prompt
    )
    conversation.append(
        {
            "role": "user",
            "content": f"Question:\n{question}\nOptions:\n{options}"
        }
    )


    text_input = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    encoding = tokenizer(text_input, return_tensors="pt").to("cpu")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            max_length=100,
            pad_token_id=tokenizer.pad_token_id,

            # generation_config = generation_config,
        )


    predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_answer = predicted_answer.split('Answer:')[-1].strip()

    # predicted_answer = predicted_answer.split('<|end_header_id|>')[-1]
    # predicted_answer = predicted_answer.replace('<|eot_id|>', '')

    return predicted_answer






def compare_answers(actual_answer: str, predicted_answer: str) -> int:
    """
    Compare the actual answer with the predicted answer.
    
    Parameters:
    - actual_answer (str): The correct answer.
    - predicted_answer (str): The answer predicted by the model.
    
    Returns:
    - int: 1 if the answers match, otherwise 0.
    """
    
    # return 1 if actual_answer.lower() == predicted_answer.lower() else 0 # v1
    
    # print("actual_answer:", actual_answer)
    # print("predicted_answer:", predicted_answer)
    

    if pd.notna(predicted_answer) and isinstance(predicted_answer, str) and predicted_answer.strip():
        matched_predicted = re.match(r'[^A-Z]*([A-Z])', predicted_answer) # v2
    else:
        return 0

    if predicted_answer.lower() == "answer":
        return 0

    if matched_predicted and matched_predicted.group(1) is not None and matched_predicted.group(1) != "":
        # print("matched_predicted:", matched_predicted.group(1).lower())
        # print('\n\n')
    
        return 1 if actual_answer.lower() == matched_predicted.group(1).lower() else 0
    else:
        return 0










# ------------------------------



def dynamic_multiple_choice_base_prompt(dataset, few_shot=5, dstype='mc'):
    # Limit to the specified number of examples
    few_shot_examples = dataset[:few_shot]
    
    messages = [
        {
            "role": "system",
            "content": "You are an AI that selects the most accurate answer in Azerbaijani based on a given question. You will be provided with a question in Azerbaijani and multiple options in Azerbaijani. Choose the single letter (A, B, C, D, ...) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
        }
    ]

    # Adding each question and options
    for entry in few_shot_examples:
        question = entry["question"]
        
        options = format_options(entry["options"], dstype)

        messages.append({
            "role": "user",
            "content": f"Question:\n{question}\nOptions:\n{options}\n\nAnswer:"
        })
        messages.append({
            "role": "assistant",
            "content": entry["correct_answer"]
        })

    return messages



