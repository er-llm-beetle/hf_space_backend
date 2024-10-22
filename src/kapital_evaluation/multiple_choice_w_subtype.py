from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import re, httpx
import pandas as pd

# from src.kapital_evaluation.utility import format_options
from utility import format_options
from dotenv import load_dotenv
from openai import OpenAI


def get_answer_multiple_choice_w_subtype(question, options, model, tokenizer, num_fewshot, dstype, base_prompt, api=False, gguf=False, repo_id=None):

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




    def generate_answer(conversation, tokenizer, model):

        # Step 1: Prepare the conversation text using the tokenizer
        text_input = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        # print("\ntext_input:", text_input, '\n')

        # Step 2: Tokenize the input text
        encoding = tokenizer(text_input, return_tensors="pt").to("cpu")

        # Step 3: Generate the output from the model
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                max_new_tokens=1000, 
                pad_token_id=tokenizer.pad_token_id,
            )

        # Step 4: Decode the output tokens into human-readable text
        # predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = tokenizer.decode(outputs[0])
        print("\npredicted_answer_1 :", predicted_answer, '\n')

        # # Step 5: Extract answer (assuming the model is generating responses with "Answer:" prefix)
        # # We handle edge cases if the "Answer:" isn't found or there are unexpected formatting issues.
        # if 'Answer:' in predicted_answer:
        #     predicted_answer = predicted_answer.split('Answer:')[-1].strip()
        
        # # Optionally handle special tokens, like <|end_header_id|> or <|eot_id|>
        # # These tokens might not always appear, so we include some checks
        # predicted_answer = predicted_answer.replace('<|end_header_id|>', '').replace('<|eot_id|>', '')



        # REGULAR EXPRESSION VERSION:
        # Use regular expression to extract everything before the last occurrence of "<|eot_id|>"
        # text_before_last_eot = re.findall(r'(.*)<\|eot_id\|>', predicted_answer)[-1]

        # v2:(check it)
        # text_before_last_eot = re.findall(r'(.*)<\|(eot_id|im_end|finish|endoftext)\|>', predicted_answer)[-1]
        # text_before_last_eot = [match[0] for match in re.findall(r'(.*)<\|(eot_id|im_end|finish|endoftext)\|>', predicted_answer)][-1]


        # # Step 6: Print and return the final answer
        # print("\npredicted_answer_2 :", text_before_last_eot, '\n')
        # return text_before_last_eot

        # v3:
        # Extract only the part before the last `<|eot_id|>` tag and return the first part of the tuple
        match = re.findall(r'(.*)<\|(eot_id|im_end|finish|endoftext)\|>', predicted_answer)
        
        # Assuming you want the first part of the tuple (e.g., 'C') from the result
        if match:
            text_before_last_eot = match[-1][0]  # Get the first part (the answer letter)
        else:
            text_before_last_eot = ''  # No match found
        
        # Step 6: Print and return the final answer
        print("\npredicted_answer_2 :", text_before_last_eot, '\n')
        return text_before_last_eot



    def generate_answer_from_api(conversation, model):
        
        load_dotenv()
    
        httpx_client = httpx.Client(http2=True, verify=False)
        
        OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

        # Initialize client for GPT 
        client_openai = OpenAI(
            # base_url=BASE_URL_LLM,
            api_key=OPENAI_API_KEY,
            http_client=httpx_client
        )

        response = client_openai.chat.completions.create(
            model=model,
            messages=conversation,
            # max_tokens=500,  
            temperature=0,  
        )

        answer = response.choices[0].message.content

        return answer


    def generate_answer_from_gguf(messages, model, repo_id):
        from llama_cpp import Llama
        # import atexit

        def main():
            llm = Llama.from_pretrained(
                # repo_id="gaianet/Nemotron-Mini-4B-Instruct-GGUF",
                # filename="Nemotron-Mini-4B-Instruct-Q2_K.gguf",
                # # filename="Nemotron-Mini-4B-Instruct-Q4_0.gguf",

                repo_id = repo_id,
                filename = model,
                n_ctx=2048,
            )

            res = llm.create_chat_completion(
                messages = messages,
                temperature=0,
                # max_tokens=300,
            )

             
            # Extract the content from the response (checking that it's available)
            try:
                content = res['choices'][0]['message']['content']
                print("content:", content)
    
                print("GGUF Answer:", content)
            except KeyError:
                print("Error: Unable to extract content from the response.")

            print(f"repoId: {repo_id} and model: {model}")
            return content

        return main()



    # return generate_answer(messages, tokenizer, model) if not api else  generate_answer_from_api(messages, model) # for api and hf version
    # return main().generate_answer_from_gguf(messages, model, repo_id) if gguf else generate_answer(messages, tokenizer, model) if not api else generate_answer_from_api(messages, model) # for api and hf and gguf versions

    # Return v3: # for api and hf and gguf versions 
    if gguf:
        print("\n\nGenerating answer using GGUF...")
        return generate_answer_from_gguf(conversation, model, repo_id)
    elif api:
        print("\n\nGenerating answer using API...")
        return generate_answer_from_api(conversation, model)
    else:
        print("\n\nGenerating answer using HF Tokenizer...")
        return generate_answer(conversation, tokenizer, model)






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
    
    print("actual_answer:", actual_answer)
    print("predicted_answer:", predicted_answer)
    

    if pd.notna(predicted_answer) and isinstance(predicted_answer, str) and predicted_answer.strip():
        matched_predicted = re.match(r'[^A-Z]*([A-Z])', predicted_answer) # v2
    else:
        return 0

    if predicted_answer.lower() == "answer":
        return 0

    if matched_predicted and matched_predicted.group(1) is not None and matched_predicted.group(1) != "":
        print("matched_predicted:", matched_predicted.group(1).lower())
        print('\n\n')
    
        return 1 if actual_answer.lower() == matched_predicted.group(1).lower() else 0
    else:
        return 0











# ------------------------------
"""


def dynamic_multiple_choice_subtype_base_prompt(dataset, few_shot=5, subtype_text='You are an AI designed to answer questions in Azerbaijani. You are an AI tasked with selecting the most accurate answer in Azerbaijani based on a given question. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.', dstype='mc'):
    
    print("dataset base:", dataset)

    # Limit to the specified number of examples
    few_shot_examples = dataset[:few_shot]
    
    # Initialize the messages list with system instruction
    messages = [
        {
            "role": "system",
            "content": subtype_text
        }
    ]

    # Adding each question and options to the prompt
    for entry in few_shot_examples:
        question = entry["question"]
        options = entry["options"]
        correct_answer = entry["answer"]
        
        # Ensure the format of the options is correct
        formatted_options = format_options(options, dstype)
        
        # Append the user prompt and the assistant's correct answer
        messages.append({
            "role": "user",
            "content": f"Question:\n{question}\nOptions:\n{formatted_options}\n\nAnswer:"
        })
        messages.append({
            "role": "assistant",
            "content": correct_answer
        })

    return messages

"""



def dynamic_multiple_choice_subtype_base_prompt(dataset, few_shot=5, subtype_text='You are an AI designed to answer questions in Azerbaijani...', dstype='mc'):
    print("dataset base:", dataset)


    # Limit to the specified number of examples
    few_shot_examples = [dataset[i] for i in range(min(few_shot, len(dataset)))]

    # Convert Hugging Face Dataset to list of dictionaries
    # dataset = dataset.to_dict(orient='records')  # If this doesn't work, try
    # dataset.to_pandas().to_dict(orient='records')

    # Limit to the specified number of examples
    # few_shot_examples = dataset[:few_shot]
    
    # Initialize the messages list with system instruction
    messages = [
        {
            "role": "system",
            "content": subtype_text
        }
    ]

    # Adding each question and options to the prompt
    for entry in few_shot_examples:
        # Debugging: print the entry to see its structure
        print(f"Entry: {entry}")
        
        # Check if entry is a dictionary and contains the expected keys
        if isinstance(entry, dict) and "question" in entry and "options" in entry and "answer" in entry:
            question = entry["question"]
            options = entry["options"]
            correct_answer = entry["answer"]
            
            # Ensure the format of the options is correct
            formatted_options = format_options(options, dstype)
            
            # Append the user prompt and the assistant's correct answer
            messages.append({
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{formatted_options}\n\nAnswer:"
            })
            messages.append({
                "role": "assistant",
                "content": correct_answer
            })
        else:
            print(f"Invalid entry format: {entry}")

    return messages
