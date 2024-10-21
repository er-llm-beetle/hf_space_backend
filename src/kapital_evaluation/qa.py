from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re, os

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import Levenshtein
from openai import OpenAI
import logging
from dotenv import load_dotenv
import httpx



# def get_answer_qa(question, model, tokenizer, device=None):
#     # Set device to GPU if available, otherwise use CPU
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     prompt = (
#         f"You are an AI designed to generate concise answers in Azerbaijani based on the following questions.\n"
#         f"Provide a clear and accurate answer in Azerbaijani, limited to 1-2 sentences and under 400 characters.\n\n"
#         f"Question in Azerbaijani:\n{question}\n\n"
#         f"Answer in Azerbaijani:"
#     )

#     # Tokenize the prompt and move tensors to the correct device
#     encoding = tokenizer(prompt, return_tensors="pt").to(device)

#     # Disable gradient calculation for inference
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=encoding.input_ids,
#             attention_mask=encoding.attention_mask,
#             max_new_tokens=300,  # Lowered max tokens for concise answers
#             pad_token_id=tokenizer.pad_token_id,
#         )

#     # Decode the predicted answer
#     predicted_answer = tokenizer.decode(outputs[0])
#     print("\npredicted_answer_qa:", predicted_answer, "\n")

#     # REGULAR EXPRESSION VERSION:
#     # Use regular expression to extract everything before the last occurrence of "<|eot_id|>"
#     text_before_last_eot = re.findall(r'(.*)<\|eot_id\|>', predicted_answer)[-1]
#     print("\ntext_before_last_eot:", text_before_last_eot, "\n\n")
    
#     return text_before_last_eot








# def get_answer_qa(question, model, tokenizer, device=None):
#     # Set device to GPU if available, otherwise use CPU
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Best practice prompt, following conventions of OpenAI, LLaMA, DeepEval, etc.
#     prompt = (
#         f"You are a knowledgeable and concise AI assistant, answering questions in Azerbaijani.\n"
#         # f"Answer the following question accurately and concisely in Azerbaijani. "
#         f"Keep the answer limited to 1-2 sentences, factual, and no more than 400 characters.\n\n"
#         f"Question: {question}\n\n"
#         f"Answer in Azerbaijani:"
#     )

#     # Tokenize the prompt and move tensors to the correct device
#     encoding = tokenizer(prompt, return_tensors="pt").to(device)

#     # Disable gradient calculation for inference
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=encoding.input_ids,
#             attention_mask=encoding.attention_mask,
#             max_new_tokens=300,  # Lowered max tokens for concise answers
#             pad_token_id=tokenizer.pad_token_id,
#         )


#     predicted_answer = tokenizer.decode(outputs[0])
#     print("\npredicted_answer_qa:", predicted_answer, "\n")

#     # REGULAR EXPRESSION VERSION:
#     # Use regular expression to extract everything before the last occurrence of "<|eot_id|>"
#     text_before_last_eot = re.findall(r'(.*)<\|eot_id\|>', predicted_answer)[-1]
#     print("\ntext_before_last_eot:", text_before_last_eot, "\n\n")
    
#     return text_before_last_eot





# Best worked for now (with 5 shot incl.):

def get_answer_qa(question, model, tokenizer=None, device=None, repo_id=None, api=False, gguf=False):
    # Set device to GPU if available, otherwise use CPU
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Few-shot messages
    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable and concise AI assistant, answering questions in Azerbaijani. "
                       "Provide factual answers that are limited to 1-2 sentences, no more than 400 characters."
        },
        {
            "role": "user",
            "content": "İqtisadi göstəricilər nədir və niyə əhəmiyyətlidir?"
        },
        {
            "role": "assistant",
            "content": "İqtisadi göstəricilər - ölkənin və ya şirkətin iqtisadi vəziyyətini qiymətləndirmək üçün istifadə olunan statistik məlumatlardır."
        },
        {
            "role": "user",
            "content": "Ekonometriya iqtisadiyyatda hansı məsələlərin həllinə kömək edir?"
        },
        {
            "role": "assistant",
            "content": "Ekonometriya iqtisadiyyatda müxtəlif məsələlərin, o cümlədən qiymət və gəlirin təyin edilməsi, istehsal və tələbin proqnozlaşdırılması, iqtisadi böhranların analizi və s. kimi məsələlərin həllinə kömək edir."
        },
        {
            "role": "user",
            "content": "Auditın məqsədi nədir?"
        },
        {
            "role": "assistant",
            "content": "Auditın məqsədi müəssisənin maliyyə hesabatlarının doğruluğunu və etibarlılığını yoxlamaqdır."
        },
        {
            "role": "user",
            "content": "Kredit kartı ilə alış-veriş etməyin faydaları nələrdir?"
        },
        {
            "role": "assistant",
            "content": "Kredit kartı ilə alış-veriş edərkən sizə müxtəlif bonuslar və endirimlər təklif olunur, bu da sizin alış-verişinizi daha sərfəli edir."
        },
        {
            "role": "user",
            "content": "Müştəri rəyləri nə üçün vacibdir?"
        },
        {
            "role": "assistant",
            "content": "Müştəri rəyləri vacibdir, çünki onlar bizə müştərilərimizin məhsullarımız və xidmətlərimiz barədə nə düşündüklərini öyrənməyə imkan verir."
        },
        # New user question
        {
            "role": "user",
            "content": question
        }
    ]


    """
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

    """


    def generate_answer(conversation, tokenizer, model):
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


    def generate_answer_from_api(conversation, model):
        
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
        return generate_answer_from_gguf(messages, model, repo_id)
    elif api:
        print("Generating answer using API...")
        return generate_answer_from_api(messages, model)
    else:
        print("Generating answer using HF Tokenizer...")
        return generate_answer(messages, tokenizer, model)







# def get_answer_qa(question, model, tokenizer, device=None):
#     # Set device to GPU if available, otherwise use CPU
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Construct the chat-style input
#     chat_template = [
#         {"role": "system", "content": "You are an AI designed to generate concise answers in Azerbaijani. Provide a clear and accurate answer in Azerbaijani, limited to 1-2 sentences and under 400 characters."},
#         {"role": "user", "content": f"Question in Azerbaijani:\n{question}\n\n"},
#         {"role": "assistant", "content": "Answer in Azerbaijani:"}
#     ]
    
#     # Convert chat template into a format that the model can use
#     prompt = tokenizer(chat_template, return_tensors="pt", padding=True).to(device)

#     # Disable gradient calculation for inference
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=prompt.input_ids,
#             attention_mask=prompt.attention_mask,
#             max_new_tokens=300,  # Limit the response length
#             pad_token_id=tokenizer.pad_token_id,
#         )

#     # Decode the predicted answer
#     predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#     print("\npredicted_answer_qa:", predicted_answer, "\n")

#     return predicted_answer












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
                    return 0
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



