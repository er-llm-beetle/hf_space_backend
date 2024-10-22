from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re, os

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import Levenshtein
from openai import OpenAI
import logging
from dotenv import load_dotenv
import httpx

# from src.kapital_evaluation.utility import format_options
# from utility import format_options





# Best worked for now (with 5 shot incl.):

def get_answer_rag_w_subtype(question, context, model, base_prompt=None, tokenizer=None, device=None, repo_id=None, api=False, gguf=False):
    # Set device to GPU if available, otherwise use CPU
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"



    conversation = []

    conversation.append(
        base_prompt
    )
    conversation.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        }
    )






    def generate_answer(messages, tokenizer, model):
        print("GENERATE ANSWER FROM HF MODEL")
        # Convert messages into a dialogue-style prompt for tokenization
        # dialogue_prompt = "".join([f"{message['role']}: {message['content']}\n" for message in messages])

        # Applying the chat template using the tokenizer's method
        dialogue_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


        # Tokenize the prompt and move tensors to the correct device
        encoding = tokenizer(dialogue_prompt, return_tensors="pt").to(device)

        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                max_new_tokens=300,  # Limit tokens for concise answers
                pad_token_id=tokenizer.pad_token_id,
            )

        predicted_answer = tokenizer.decode(outputs[0])
        print("\npredicted_answer_qa:", predicted_answer, "\n")

        # # REGULAR EXPRESSION VERSION:
        # # Use regular expression to extract everything before the last occurrence of "<|eot_id|>"
        # # text_before_last_eot = re.findall(r'(.*)<\|(?:eot_id|im_end)\|>', predicted_answer)[-1]
        # # text_before_last_eot = re.findall(r'(.*)<\|(eot_id|im_end|finish)\|>', predicted_answer)[-1]

        # v2:
        # text_before_last_eot = [match[0] for match in re.findall(r'(.*)<\|(eot_id|im_end|finish|endoftext)\|>', predicted_answer)][-1]
        # print("\ntext_before_last_eot:", text_before_last_eot, "\n\n")
        
        # return text_before_last_eot

        # v3:
        match = re.findall(r'(.*)<\|(eot_id|im_end|finish|endoftext)\|>', predicted_answer)
        
        # Assuming you want the first part of the tuple (e.g., 'C') from the result
        if match:
            text_before_last_eot = match[-1][0]  # Get the first part (the answer letter)
        else:
            text_before_last_eot = ''  # No match found
        
        # Step 6: Print and return the final answer
        print("\npredicted_answer_2 :", text_before_last_eot, '\n')
        return text_before_last_eot



    def generate_answer_from_api(messages, model):
        
        load_dotenv()
        
        OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
    
        httpx_client = httpx.Client(http2=True, verify=False)

        # Initialize client for GPT 
        client_openai = OpenAI(
            # base_url=BASE_URL_LLM,
            api_key=OPENAI_API_KEY,
            http_client=httpx_client
        )

        response = client_openai.chat.completions.create(
            model=model,
            messages=messages,
            # max_tokens=500,  
            temperature=0,  
        )

        answer = response.choices[0].message.content
        print("API Answer:", answer)

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
        print("Generating answer using GGUF...")
        return generate_answer_from_gguf(conversation, model, repo_id)
    elif api:
        print("Generating answer using API...")
        return generate_answer_from_api(conversation, model)
    else:
        print("Generating answer using HF Tokenizer...")
        return generate_answer(conversation, tokenizer, model)











def get_evaluation_score(question: str, actual_answer: str, predicted_answer: str) -> str:
    """
    Generate an evaluation score between 0 and 100 by comparing the actual and predicted answers.
    """

    # httpx_client = httpx.Client(http2=True, verify=False)

    # Initialize OpenAI client for NVIDIA
    # client = OpenAI(base_url=BASE_URL_LLM, api_key=API_KEY_LLM, http_client=httpx_client)

    load_dotenv()


    BASE_URL_LLM = "https://integrate.api.nvidia.com/v1"
    MODEL_LLAMA_3_1_405B = "meta/llama-3.1-405b-instruct"
    MODEL_LLAMA_3_1_8B = "meta/llama-3.1-8b-instruct"
    API_KEY_LLM = os.getenv('API_KEY_NVIDIA_LLM')

    httpx_client = httpx.Client(http2=True, verify=False)

    # Initialize OpenAI client for NVIDIA 405b
    client_nvidia = OpenAI(
        base_url=BASE_URL_LLM, 
        api_key=API_KEY_LLM, 
        http_client=httpx_client
    )


    # v2
    prompt = f"""
            Evaluate the following answers and provide a score from 0 to 100 based on how well the predicted
            answer matches the actual answer based on the asked question. Provide the score only, without any additional text.

            0-10: No answer or completely incorrect
            11-30: Significant errors or missing key information
            31-50: Some errors or incomplete information, but recognizable effort
            51-70: Mostly accurate with minor errors or omissions
            71-90: Very close to the actual answer with only minor discrepancies
            91-100: Accurate or nearly perfect match

            **Example:**

            **Question that asked in Azerbaijani:** Makroiqtisadiyyat nədir və mikroiqtisadiyyatdan necə fərqlənir?  
            **Actual Answer in Azerbaijani:** Makroiqtisadiyyat iqtisadiyyatın böyük miqyasda təhlili ilə məşğul olur, mikroiqtisadiyyat isə kiçik miqyasda, yəni fərdi bazarlarda və şirkətlərdə baş verən prosesləri öyrənir.  
            **Predicted Answer in Azerbaijani:** Makroiqtisadiyyat iqtisadiyyatın ümumi aspektlərini öyrənir, mikroiqtisadiyyat isə fərdi bazarları təhlil edir.  
            **Score (0 to 100):** 65

            **Your Task:**

            **Question that asked:** {question}

            **Actual Answer:** {actual_answer}

            **Predicted Answer:** {predicted_answer}

            **Score (0 to 100):**
            """


    payload = {
        "model": MODEL_LLAMA_3_1_405B,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 50
    }



    try:
        completion = client_nvidia.chat.completions.create(**payload)
        if completion.choices:
            content = completion.choices[0].message.content
            if content:
                score = content.strip()
                print("score GPT Eval:", score)
                if score.lower() == 'error':
                    return '0'
                return score

            logging.error("Content in response is None.")
        else:
            logging.error(f"Unexpected response format: {completion}")
    except Exception as e:
        logging.error(f"Request failed: {e}")
    return "Error"






def calculate_bleu_score(actual_answer: str, predicted_answer: str) -> float:
    """
    Calculate BLEU score for the given actual and predicted answers.
    """
    reference = actual_answer.split()
    candidate = predicted_answer.split()
    bleu_score = sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)
    print('bleu:', 25 * bleu_score)
    return 25 * bleu_score

def calculate_rouge_score(actual_answer: str, predicted_answer: str) -> dict:
    """
    Calculate ROUGE score for the given actual and predicted answers.
    """
    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    scores = scorer.score(actual_answer, predicted_answer)

    normalized_scores = {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

    print('rouge:', 25 * (0.33 * normalized_scores['rouge1'] + 0.33 * normalized_scores['rouge2'] + 0.33 * normalized_scores['rougeL']))
    return 25 * (0.33 * normalized_scores['rouge1'] + 0.33 * normalized_scores['rouge2'] + 0.33 * normalized_scores['rougeL'])


def calculate_levenshtein_score(actual_answer: str, predicted_answer: str) -> int:
    """
    Calculate Levenshtein distance between actual and predicted answers.
    """

    max_len = max(len(actual_answer), len(predicted_answer))

    if max_len == 0:  # both strings are empty
        return 1

    print('levenshtein:', 25 * (1 - (Levenshtein.distance(actual_answer, predicted_answer) / max_len)))
    return 25 * (1 - (Levenshtein.distance(actual_answer, predicted_answer) / max_len))










def dynamic_rag_subtype_base_prompt(dataset, few_shot=5, subtype_text='You are an AI designed to answer questions in Azerbaijani.', dstype='mc'):
    print("dataset base qa:", dataset)


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
        if isinstance(entry, dict):
            question = entry["question"]
            context = entry["context"]
            correct_answer = entry["answer"]


            {
                "role": "user",
                "content": "Auditın məqsədi nədir?"
            },
            {
                "role": "assistant",
                "content": "Auditın məqsədi müəssisənin maliyyə hesabatlarının doğruluğunu və etibarlılığını yoxlamaqdır."
            },

            messages.append({
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
            })
            messages.append({
                "role": "assistant",
                "content": correct_answer
            })
        else:
            print(f"Invalid entry format: {entry}")

    return messages




