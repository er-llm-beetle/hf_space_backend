from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re



def get_answer_rag(question, context, model, tokenizer):
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


    # Clear and structured prompt
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


