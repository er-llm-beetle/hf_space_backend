from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re, os
import httpx

from dotenv import load_dotenv
from openai import OpenAI


def get_answer_rag(question, context, model, tokenizer=None, api=False, gguf=False, repo_id=None):
    # prompt = (
    #     f"You are an answer generator AI in Azerbaijani. Your task is to generate concise answers based on the provided context and the given question.\n"
    #     f"Provide a clear and accurate answer in Azerbaijani, limited to 1-2 sentences and under 400 characters.\n\n"
    #     "### CONTEXT ###\n\n"
    #     '"""\n\n'
    #     f"{context}\n\n"
    #     '"""\n\n'
    #     "### END CONTEXT ###\n\n"
    #     f"Question: {question} Answer in Azerbaijani:\n"
    # )


    # Clear and structured prompt (Working one)
    prompt = (
        "You are a retrieval-augmented answer generator AI in Azerbaijani. "
        "Use the provided context to generate a concise and accurate answer to the question. "
        "Ensure the answer is limited to 1-2 sentences and under 400 characters.\n\n"
        "### CONTEXT ###\n"
        f"{context}\n\n"
        "### QUESTION ###\n"
        f"{question}\n\n"
        "### ANSWER ###"
    )


    # Message version
    messages = [
        {
            "role": "system", 
            "content": "You are a retrieval-augmented answer generator AI in Azerbaijani. Use the provided context to generate a concise and accurate answer to the question, limited to 1-2 sentences and under 400 characters.\n"},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        }
    ]


    """

    # Tokenize the prompt
    encoding = tokenizer(prompt, return_tensors="pt").to("cpu")

    # Disable gradient calculation for inference
    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            max_new_tokens=300,  # Corrected the argument
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode the predicted answer
    predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("\npredicted_answer_rag:", predicted_answer, "\n")

    # REGULAR EXPRESSION VERSION:
    # Use regular expression to extract everything before the last occurrence of "<|eot_id|>"
    # text_before_last_eot = re.findall(r'(.*)<\|(?:eot_id|im_end)\|>', predicted_answer)[-1]
    # text_before_last_eot = re.findall(r'(.*)<\|(eot_id|im_end|finish)\|>', predicted_answer)[-1]

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
 
        # Applying the chat template using the tokenizer's method
        # dialogue_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


        # Tokenize the prompt
        encoding = tokenizer(prompt, return_tensors="pt").to("cpu")

        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                max_new_tokens=300,  # Corrected the argument
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode the predicted answer
        predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print("\npredicted_answer_rag:", predicted_answer, "\n")

        # REGULAR EXPRESSION VERSION:
        # Use regular expression to extract everything before the last occurrence of "<|eot_id|>"
        # text_before_last_eot = re.findall(r'(.*)<\|(?:eot_id|im_end)\|>', predicted_answer)[-1]
        # text_before_last_eot = re.findall(r'(.*)<\|(eot_id|im_end|finish)\|>', predicted_answer)[-1]

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
                messages=messages,
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
        print("\nGenerating answer using GGUF...")
        return generate_answer_from_gguf(messages, model, repo_id)
    elif api:
        print("\nGenerating answer using API...")
        # return generate_answer_from_api(prompt, model)
        return generate_answer_from_api(messages, model)
    else:
        print("\nGenerating answer using HF Tokenizer...")
        # return generate_answer(prompt, tokenizer, model)
        return generate_answer(messages, tokenizer, model)
