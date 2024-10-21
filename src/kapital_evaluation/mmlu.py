# from datasets import load_dataset
# from transformers import AutoModelForCausalLM
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Use Seq2SeqLM for T5
# from tqdm import tqdm
# import torch

# # from src.kapital_evaluation.qa import get_answer_qa, calculate_bleu_score, calculate_rouge_score, calculate_levenshtein_score
# # from src.kapital_evaluation.rag import get_answer_rag
# # from src.kapital_evaluation.multiple_choice_w_dstype_main import get_answer_multiple_choice_w_dstype, compare_answers
# # # from src.kapital_evaluation.multiple_choice_w_subtype import get_answer_multiple_choice_w_subtype, dynamic_multiple_choice_subtype_base_prompt



# from qa import get_answer_qa, calculate_bleu_score, calculate_rouge_score, calculate_levenshtein_score
# from rag import get_answer_rag
# from multiple_choice_w_dstype_main import get_answer_multiple_choice_w_dstype, compare_answers
# # from src.kapital_evaluation.multiple_choice_w_subtype import get_answer_multiple_choice_w_subtype, dynamic_multiple_choice_subtype_base_prompt

# from detect_model import detect_and_print_model_info


# def handle_qa_score(actual_answer, predicted_answer):
#     if predicted_answer.lower() == 'long answer' or 'error' in predicted_answer.lower():
#         return 0
    
#     score = (
#         calculate_bleu_score(actual_answer, predicted_answer) \
#         + calculate_rouge_score(actual_answer, predicted_answer) \
#         + calculate_levenshtein_score(actual_answer, predicted_answer)
#     )
#     return score

# #  err => pydantic-2.9.2 pydantic-core-2.23.4

# def handle_context_qa_score(actual_answer, predicted_answer):
#     if predicted_answer.lower() == 'long answer' or 'error' in predicted_answer.lower():
#         return 0

#     score = (
#         calculate_bleu_score(actual_answer, predicted_answer) \
#         + calculate_rouge_score(actual_answer, predicted_answer) \
#         + calculate_levenshtein_score(actual_answer, predicted_answer)
#     )
#     return score

# # def compare_answers(actual_answer: str, predicted_answer: str) -> int:
# #     return actual_answer.strip().lower() == predicted_answer.strip().lower()

# def evaluate(model, dtype, tasks, num_fewshot, batch_size, device, limit=2, write_out=True):



#     model_name = model

#     # result = {
#     #     "config": {},
#     #     "results": {
#     #     }
#     # }

#     # tokenizer = AutoTokenizer.from_pretrained(model, cache_dir='cache')
#     # model = AutoModelForSeq2SeqLM.from_pretrained(model, cache_dir='cache').to("cpu")  # Changed to Seq2SeqLM
    

#     # Directory where you saved the model
#     # save_directory = "/Users/rahimovamir/Downloads/huggingface_llm/OpenLLM-Azerbaijani-Backend/llama_3.2_1b_instruct_model"
#     # save_directory = "/Users/rahimovamir/Downloads/huggingface_llm/OpenLLM-Azerbaijani-Backend/Qwen2.5-0.5B-Instruct_model"
#     save_directory = "/Users/rahimovamir/Downloads/huggingface_llm/OpenLLM-Azerbaijani-Backend/Phi-3.5-mini-instruct_model"
    
#     # Load the model and tokenizer from the local directory
#     # model = AutoModelForCausalLM.from_pretrained(save_directory, cache_dir='cache')
#     # tokenizer = AutoTokenizer.from_pretrained(save_directory, cache_dir='cache')

#     print(f"Model and tokenizer loaded from {save_directory}")



#     tokenizer, model = detect_and_print_model_info(save_directory)

#     print(f"Loaded model: {model_name}")
#     # print(f"Tokenizer: {tokenizer}")
#     print(f"Model: {model}")




#     # model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='cache')
#     # tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='cache') #.to("cpu") 

#     # generation_config = model.generation_config
#     # generation_config.max_length = 1000
#     # generation_config.pad_token_id = tokenizer.pad_token_id
    


#     # v4 with dstype and w subtext
#     datasets = [
#         # {
#         #     "task_type": "mmlu",
#         #     "dstype": "mc",
#         #     "group": "banking",
#         #     "subtext": "You are an AI that selects the most accurate answer in Azerbaijani based on a given question. You will be provided with a question in Azerbaijani and multiple options in Azerbaijani. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
#         #     "data": load_dataset("LLM-Beetle/Banking-benchmark_mmlu_fqa_latest")["train"],
#         # },
        
#         # {
#         #     "task_type": "mmlu",
#         #     "dstype": "kmc_azerbaycan_dili",
#         #     "group": "mmlu",
#         #     "subtext": "You are an AI designed to answer questions in Azerbaijani based on grammatical concepts and linguistics. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
#         #     # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
#         #     "data": load_dataset("Emirrv/mmlu_aze-testler_azerbaycan_dili_kmc")["train"],
#         # },
#         # {
#         #     "task_type": "mmlu",
#         #     "dstype": "kmc_edebiyyat",
#         #     "group": "mmlu",
#         #     "subtext": "You are an AI designed to answer questions in Azerbaijani based on literary and historical facts. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
#         #     # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
#         #     "data": load_dataset("Emirrv/mmlu_aze-testler_edebiyyat_kmc")["train"]
#         # },
#         # {
#         #     "task_type": "mmlu",
#         #     "dstype": "kmc_biologiya",
#         #     "group": "mmlu",
#         #     "subtext": "You are an AI designed to answer questions in Azerbaijani based on biology. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
#         #     # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
#         #     "data": load_dataset("Emirrv/mmlu_aze-testler_biologiya_kmc")["train"]
#         # },
#         # {
#         #     "task_type": "mmlu",
#         #     "dstype": "kmc_cografiya",
#         #     "group": "mmlu",
#         #     "subtext": "You are an AI designed to answer questions in Azerbaijani based on geographical and environmental knowledge. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
#         #     # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
#         #     "data": load_dataset("Emirrv/mmlu_aze-testler_cografiya_kmc")["train"]
#         # },
#         # {
#         #     "task_type": "mmlu",
#         #     "dstype": "kmc_mentiq",
#         #     "group": "mmlu",
#         #     "subtext": "You are an AI designed to answer questions in Azerbaijani based on logical reasoning and problem-solving. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
#         #     # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
#         #     "data": load_dataset("Emirrv/mmlu_aze-testler_mentiq_kmc")["train"]
#         # },
#         # {
#         #     "task_type": "mmlu",
#         #     "dstype": "kmc_tarix",
#         #     "group": "mmlu",
#         #     "subtext": "You are an AI designed to answer questions in Azerbaijani based on historical and cultural facts. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
#         #     # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
#         #     "data": load_dataset("Emirrv/mmlu_aze-testler_tarix_kmc")["train"]
#         # },
#         # {
#         #     "task_type": "mmlu",
#         #     "dstype": "kmc_informatika",
#         #     "group": "mmlu",
#         #     "subtext": "You are an AI designed to answer questions in Azerbaijani based on technology and computer science. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
#         #     # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
#         #     "data": load_dataset("Emirrv/mmlu_aze-testler_informatika_kmc")["train"]
#         # },
#         # {
#         #     "task_type": "mmlu",
#         #     "dstype": "kmc_fizika",
#         #     "group": "mmlu",
#         #     "subtext": "You are an AI designed to answer questions in Azerbaijani based on physics concepts. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
#         #     # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
#         #     "data": load_dataset("Emirrv/mmlu_aze-testler_fizika_kmc")["train"]
#         # },
#         # {
#         #     "task_type": "mmlu",
#         #     "dstype": "kmc_kimya",
#         #     "group": "mmlu",
#         #     "subtext": "You are an AI designed to answer questions in Azerbaijani based on chemistry and scientific concepts. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
#         #     # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
#         #     "data": load_dataset("Emirrv/mmlu_aze-testler_kimya_kmc")["train"]
#         # },
#         # {
#         #     "task_type": "mmlu",
#         #     "dstype": "kmc_azerbaycan_tarixi",
#         #     "group": "mmlu",
#         #     "subtext": "You are an AI designed to answer questions in Azerbaijani based on historical facts. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
#         #     # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
#         #     "data": load_dataset("Emirrv/mmlu_aze-testler_azerbaycan_tarixi_kmc")["train"]
#         # },
 
#         # {
#         #     "task_type": "mmlu",
#         #     "dstype": "tc",
#         #     "group": "banking",
#         #     "subtext": "You are an AI designed to answer questions in Azerbaijani. Your task is to select the correct option from the given question and answer choices. You are given a statement along with multiple options that represent different topics. Choose the option that best categorizes the statement based on its topic. Choose the single letter (A, B, C, D, E, F, G, H, I, J) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
#         #     # "data": load_dataset("LLM-Beetle/banking_support")["train"]
#         #     "data": load_dataset("Emirrv/Banking_support_aze_version_reshad_tc")["train"]
#         # },
#         # {
#         #     "task_type": "mmlu",
#         #     "dstype": "arc",
#         #     "group": "arc",
#         #     "subtext": "You are an AI designed to answer questions in Azerbaijani based on reasoning and knowledge. Your task is to select the correct option from the given question and answer choices. You are given a question along with multiple options. Choose the correct option. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",  # Reshad's update
#         #     # "data": load_dataset("LLM-Beetle/arc_translated")["train"],
#         #     # "data": load_dataset("Emirrv/arc_translated_cutted_with_extra_cols_and_options_structure_last_version_mmlu_arc")["train"],
#         #     "data": load_dataset("Emirrv/arc_translated_cutted_with_new_aze_option_structure_latest_mmlu_arc")["train"], # simplified, aze version (question, answer, options column )
#         # },

#         # {
#         #     "task_type": "mmlu",
#         #     "dstype": "mmc",
#         #     "group": "math",
#         #     "subtext": "You are an AI designed to solve mathematical word problems in Azerbaijani. Your task is to analyze the given question and select the correct option from the provided choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
#         #     "data": load_dataset("Emirrv/gsm8k_mmc")["train"],
#         # },

#         # {
#         #     "task_type": "mmlu_context",
#         #     "group": "mmlu",
#             # "dstype": "qmc",
#         #     "data": load_dataset("LLM-Beetle/anl_quad")["train"],
#         # },
#         # {
#         #     # "task_type": "mmlu_context_check", ??
#         #     "group": "context",
#             # "dstype": "",
#         #     "data": load_dataset("LLM-Beetle/fact_checker")["train"]
#         # },
#         {
#             "task_type": "qa",
#             "dstype": "qa",
#             "group": "economics",
#             "subtext": "",
#             "data": load_dataset("LLM-Beetle/LLM_generated_qa_latest")["train"]
#         },
#         # {
#         #     "task_type": "rag",
#         #     "dstype": "cqa",
#         #     "group": "context",
#         #     "subtext": "",
#         #     "data": load_dataset("LLM-Beetle/Quad_benchmark_cqa_latest")["train"]
#         # },
#     ]



#     result = {
#         "config": {
#             # "num_fewshot": num_fewshot,
#             # "batch_size": batch_size,
#             # "device": device,
#             # "limit": limit,
#         },
#         # "dataset_scores": {},
#         # "group_scores": {},
#         "results": {},

#     }


#     print("datasets", datasets)


#     for dataset in datasets:
#         task_type = dataset['task_type']
#         data = dataset['data']
#         dstype = dataset['dstype']
#         group = dataset['group']

#         total_score = 0
#         limit = 2

#         print("dataset:", dataset)
#         print("dstype:", dstype)
#         print("tasktype:", task_type)

#         # limit = limit or 0  # Sets `limit` to 0 if it's None

#         if data:
#             total_limit = min(limit, len(data))
#         else:
#             raise ValueError('errror')

#         print("total_limit", total_limit)

#         dataset_score = 0
#         correct_count = 0


#         # base_prompt_mmlu = dynamic_multiple_choice_base_prompt(dataset=data, few_shot=5)
#         # base_prompt_subtype_mmlu = dynamic_multiple_choice_subtype_base_prompt(
#             # dataset=data, few_shot=5
#         # )

#         for i in tqdm(range(total_limit), desc=f'Evaluating task_type: {task_type} - dstype: {dstype}'):
#             # question = data['question'][i] if data['question'][i] else None
#             # if question == None and dstype == 'arc':
#             #     question = data['Azerbaijani_q'][i]

#             # correct_answer = data['answer'][i]

#             # options = data['options'][i] if task_type == "mmlu" and data['options'][i] else None
#             # if options == None:
#             #     options = data['choices'][i]

#             # context = data['context'][i] if task_type in ["rag", "qa"] else None

#             # Retrieve the question column dynamically based on the dataset and dstype
#             if 'question' in data.column_names:
#                 question = data['question'][i]
#             # elif dstype == 'arc' and 'Azerbaijani_q' in data.column_names:
#             #     question = data['Azerbaijani_q'][i]
#             # else:
#             #     raise KeyError("Neither 'question' nor 'Azerbaijani_q' found in the dataset.")

#             # Retrieve the correct answer
#             if 'answer' in data.column_names:
#                 correct_answer = data['answer'][i]
#             # elif dstype == 'arc' and 'answerKey' in data.column_names:
#             #     correct_answer = data['answerKey'][i]


#             # Dynamically select the options column based on what exists
#             if 'options' in data.column_names:
#                 options = data['options'][i]
#             elif 'choices' in data.column_names:
#                 options = data['choices'][i]
#             elif task_type in ['qa', 'rag']:
#                 pass
#             else:
#                 raise KeyError("Neither 'options' nor 'choices' found in the dataset.")

#             # Handle the context column if applicable
#             context = data['context'][i] if task_type in ["rag"] else None




#             if task_type == "mmlu":
#                 predicted_answer = get_answer_multiple_choice_w_dstype(
#                     question=question, options=options, model=model, num_fewshot=0, dstype=dstype, tokenizer=tokenizer 
#                 )
#             elif task_type == "qa":
#                 print('\n\n TASK TYPE - QA: \n')
#                 # predicted_answer = get_answer_qa(question, model)  # Removed tokenizer
#                 predicted_answer = get_answer_qa(question, model, tokenizer)
#             elif task_type == "rag":
#                 # predicted_answer = get_answer_rag(question, context, model)  # Removed tokenizer
#                 predicted_answer = get_answer_rag(question, context, model, tokenizer)
#             else:
#                 raise ValueError("Invalid task type")

#             if task_type in ["mmlu", "mmlu_context"]:
#                 score = compare_answers(correct_answer, predicted_answer)
#                 print("\n\nscore MMLU:", score, "\n")
#                 total_score += score
#                 dataset_score += score
#                 print("\ntotal_score:", total_score, "\n")
#                 print("\ndataset_score:", dataset_score, "\n\n")

#             elif task_type == "qa":
#                 score = handle_qa_score(correct_answer, predicted_answer)
#                 print("\n\nscore QA:", score, "\n")
#                 total_score += score
#                 dataset_score += score
#                 print("\ntotal_score:", total_score, "\n")
#                 print("\ndataset_score:", dataset_score, "\n\n")

#             elif task_type == "rag":
#                 score = handle_context_qa_score(correct_answer, predicted_answer)
#                 print("\n\nscore RAG:", score, "\n")
#                 total_score += score
#                 dataset_score += score
#                 print("\ntotal_score:", total_score, "\n")
#                 print("\ndataset_score:", dataset_score, "\n\n")



#             # SCORES:  
#             # Version 1 (base version)
#             # if task_type in ["mmlu", "mmlu_context"]:
#             #     score = compare_answers(correct_answer, predicted_answer)
#             #     print("\n\nscore:", score, "\n")
#             #     total_score += score
#             #     print("\ntotal_score:", score, "\n\n")

#             # elif task_type == "qa":
#             #     score = handle_qa_score(actual_answer=correct_answer, predicted_answer=predicted_answer)
#             #     # score = handle_qa_score(question=question, actual_answer=correct_answer, predicted_answer=predicted_answer)
#             #     total_score += score

#             # elif task_type == "rag":
#             #     score = handle_context_qa_score(actual_answer=correct_answer, predicted_answer=predicted_answer)
#             #     # score = handle_qa_score(question=question, actual_answer=correct_answer, predicted_answer=predicted_answer)
#             #     total_score += score
        
#         result["results"][dstype] = {
#             "metric_name": total_score / total_limit if total_limit > 0 else 0.0
#         }    

#         print("\n\n", result["results"], "\n\n")
#         # result.save_to_disk('path/to/save/') # save
#         # result.to_parquet('path/to/save/dataset.parquet')


# # 
#     #     average_dataset_score = dataset_score / total_limit if total_limit > 0 else 0.0
#     #     result["dataset_scores"][f"{group}_{task_type}"] = average_dataset_score

#     #     if group not in result["group_scores"]:
#     #         result["group_scores"][group] = {
#     #             "total_score": 0,
#     #             "total_count": 0
#     #         }
#     #     result["group_scores"][group]["total_score"] += correct_count
#     #     result["group_scores"][group]["total_count"] += total_limit

#     # for group, scores in result["group_scores"].items():
#     #     total_score = scores["total_score"]
#     #     total_count = scores["total_count"]
#     #     result["group_scores"][group] = total_score / total_count if total_count > 0 else 0.0



#     return result


# if __name__ == '__main__':
#     # result = evaluate('LLM-Beetle/ProdataAI_Llama-3.1-8bit-Instruct', 5, 32, 'cpu', 10)
#     # print(result)

#     # print(evaluate('LLM-Beetle/ProdataAI_Llama-3.1-8bit-Instruct',0,0,0,0,0,2,0))
#     print(evaluate('LLM-Beetle/ProdataAI_Llama-3.1-8bit-Instruct',0,0,0,0,0,2,0))






































# ------------------ w API version (with some additions) ---------------------------







from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Use Seq2SeqLM for T5
from tqdm import tqdm
import torch

# For general:

# from src.kapital_evaluation.qa import get_answer_qa, calculate_bleu_score, calculate_rouge_score, calculate_levenshtein_score, get_evaluation_score
# from src.kapital_evaluation.rag import get_answer_rag
# from src.kapital_evaluation.multiple_choice_w_dstype_main import get_answer_multiple_choice_w_dstype, compare_answers
# # from src.kapital_evaluation.multiple_choice_w_subtype import get_answer_multiple_choice_w_subtype, dynamic_multiple_choice_subtype_base_prompt
# from src.kapital_evaluation.detect_model import detect_and_print_model_info


# For local:

from qa import get_answer_qa, calculate_bleu_score, calculate_rouge_score, calculate_levenshtein_score, get_evaluation_score
from rag import get_answer_rag
from multiple_choice_w_dstype_main import get_answer_multiple_choice_w_dstype, compare_answers
from detect_model import detect_and_print_model_info



def handle_qa_score(actual_answer, predicted_answer, question):
    if predicted_answer.lower() == 'long answer' or 'error' in predicted_answer.lower():
        return 0
    
    score = (
        calculate_bleu_score(actual_answer, predicted_answer) \
        + calculate_rouge_score(actual_answer, predicted_answer) \
        + calculate_levenshtein_score(actual_answer, predicted_answer) \
        + 0.25 * int(float(get_evaluation_score(question, actual_answer, predicted_answer)))
    )
    return score

#  err => pydantic-2.9.2 pydantic-core-2.23.4

def handle_context_qa_score(actual_answer, predicted_answer, question):
    if predicted_answer.lower() == 'long answer' or 'error' in predicted_answer.lower():
        return 0

    score = (
        calculate_bleu_score(actual_answer, predicted_answer) \
        + calculate_rouge_score(actual_answer, predicted_answer) \
        + calculate_levenshtein_score(actual_answer, predicted_answer) \
        + 0.25 * int(float(get_evaluation_score(actual_answer, predicted_answer, question)))
    )
    return score

# def compare_answers(actual_answer: str, predicted_answer: str) -> int:
#     return actual_answer.strip().lower() == predicted_answer.strip().lower()





def evaluate(model, dtype, tasks, num_fewshot, batch_size, device, limit=2, write_out=True):

    # Hard coded for now
    API = False
    GGUF = False

    # FIX IT FOR REAL CASE
    model_name = model
    

    # result = {
    #     "config": {},
    #     "results": {
    #     }
    # }

    if not API and not GGUF:

        # tokenizer = AutoTokenizer.from_pretrained(model, cache_dir='cache')
        # model = AutoModelForSeq2SeqLM.from_pretrained(model, cache_dir='cache').to("cpu")  # Changed to Seq2SeqLM
        

        # Directory where you saved the model (for local main testing)
        save_directory = "/Users/rahimovamir/Downloads/huggingface_llm/OpenLLM-Azerbaijani-Backend/llama_3.2_1b_instruct_model"
        # save_directory = "/Users/rahimovamir/Downloads/huggingface_llm/OpenLLM-Azerbaijani-Backend/Qwen2.5-0.5B-Instruct_model"
        # save_directory = "/Users/rahimovamir/Downloads/huggingface_llm/OpenLLM-Azerbaijani-Backend/Phi-3.5-mini-instruct_model"
        # save_directory = "/Users/rahimovamir/Downloads/huggingface_llm/OpenLLM-Azerbaijani-Backend/Llama3.1-elm-turbo-3B-instruct_model"

        # save_directory = model_name

        # Load the model and tokenizer from the local directory
        # model = AutoModelForCausalLM.from_pretrained(save_directory, cache_dir='cache')
        # tokenizer = AutoTokenizer.from_pretrained(save_directory, cache_dir='cache')

        print(f"Model and tokenizer loaded from {save_directory}")



        tokenizer, model = detect_and_print_model_info(save_directory)

        print(f"Loaded model: {model_name}")
        # print(f"Tokenizer: {tokenizer}")
        print(f"Model: {model}")


        print(f"Starting HF Tokenizer Version for model: {model}")
        


        # model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='cache')
        # tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='cache') #.to("cpu") 

        # generation_config = model.generation_config
        # generation_config.max_length = 1000
        # generation_config.pad_token_id = tokenizer.pad_token_id
    
    elif API:
        model = 'gpt-4o-mini'
        print(f"Starting API Version for model: {model}")

    elif GGUF:
        repo_id = 'bartowski/Llama-3.2-3B-Instruct-GGUF'
        model = 'Llama-3.2-3B-Instruct-Q4_0.gguf'
        # model = model

        # repo_id = ''
        # model = 'unsloth.Q4_K_M.gguf'

        print(f"Starting GGUF Version for model: {model}")


    # Metadata:

    # v4 with dstype and w subtext
    datasets = [
        {
            "task_type": "mmlu",
            "dstype": "mc",
            "group": "banking",
            "subtext": "You are an AI that selects the most accurate answer in Azerbaijani based on a given question. You will be provided with a question in Azerbaijani and multiple options in Azerbaijani. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
            "data": load_dataset("LLM-Beetle/Banking-benchmark_mmlu_fqa_latest")["train"],
        },
        
        {
            "task_type": "mmlu",
            "dstype": "kmc_azerbaycan_dili",
            "group": "mmlu",
            "subtext": "You are an AI designed to answer questions in Azerbaijani based on grammatical concepts and linguistics. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
            # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
            "data": load_dataset("Emirrv/mmlu_aze-testler_azerbaycan_dili_kmc")["train"],
        },
        {
            "task_type": "mmlu",
            "dstype": "kmc_edebiyyat",
            "group": "mmlu",
            "subtext": "You are an AI designed to answer questions in Azerbaijani based on literary and historical facts. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
            # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
            "data": load_dataset("Emirrv/mmlu_aze-testler_edebiyyat_kmc")["train"]
        },
        {
            "task_type": "mmlu",
            "dstype": "kmc_biologiya",
            "group": "mmlu",
            "subtext": "You are an AI designed to answer questions in Azerbaijani based on biology. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
            # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
            "data": load_dataset("Emirrv/mmlu_aze-testler_biologiya_kmc")["train"]
        },
        {
            "task_type": "mmlu",
            "dstype": "kmc_cografiya",
            "group": "mmlu",
            "subtext": "You are an AI designed to answer questions in Azerbaijani based on geographical and environmental knowledge. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
            # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
            "data": load_dataset("Emirrv/mmlu_aze-testler_cografiya_kmc")["train"]
        },
        {
            "task_type": "mmlu",
            "dstype": "kmc_mentiq",
            "group": "mmlu",
            "subtext": "You are an AI designed to answer questions in Azerbaijani based on logical reasoning and problem-solving. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
            # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
            "data": load_dataset("Emirrv/mmlu_aze-testler_mentiq_kmc")["train"]
        },
        {
            "task_type": "mmlu",
            "dstype": "kmc_tarix",
            "group": "mmlu",
            "subtext": "You are an AI designed to answer questions in Azerbaijani based on historical and cultural facts. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
            # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
            "data": load_dataset("Emirrv/mmlu_aze-testler_tarix_kmc")["train"]
        },
        {
            "task_type": "mmlu",
            "dstype": "kmc_informatika",
            "group": "mmlu",
            "subtext": "You are an AI designed to answer questions in Azerbaijani based on technology and computer science. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
            # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
            "data": load_dataset("Emirrv/mmlu_aze-testler_informatika_kmc")["train"]
        },
        {
            "task_type": "mmlu",
            "dstype": "kmc_fizika",
            "group": "mmlu",
            "subtext": "You are an AI designed to answer questions in Azerbaijani based on physics concepts. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
            # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
            "data": load_dataset("Emirrv/mmlu_aze-testler_fizika_kmc")["train"]
        },
        {
            "task_type": "mmlu",
            "dstype": "kmc_kimya",
            "group": "mmlu",
            "subtext": "You are an AI designed to answer questions in Azerbaijani based on chemistry and scientific concepts. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
            # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
            "data": load_dataset("Emirrv/mmlu_aze-testler_kimya_kmc")["train"]
        },
        {
            "task_type": "mmlu",
            "dstype": "kmc_azerbaycan_tarixi",
            "group": "mmlu",
            "subtext": "You are an AI designed to answer questions in Azerbaijani based on historical facts. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
            # "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
            "data": load_dataset("Emirrv/mmlu_aze-testler_azerbaycan_tarixi_kmc")["train"]
        },
 
        {
            "task_type": "mmlu",
            "dstype": "tc",
            "group": "banking",
            "subtext": "You are an AI designed to answer questions in Azerbaijani. Your task is to select the correct option from the given question and answer choices. You are given a statement along with multiple options that represent different topics. Choose the option that best categorizes the statement based on its topic. Choose the single letter (A, B, C, D, E, F, G, H, I, J) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
            # "data": load_dataset("LLM-Beetle/banking_support")["train"]
            "data": load_dataset("Emirrv/Banking_support_aze_version_reshad_tc")["train"]
        },
        {
            "task_type": "mmlu",
            "dstype": "arc",
            "group": "arc",
            "subtext": "You are an AI designed to answer questions in Azerbaijani based on reasoning and knowledge. Your task is to select the correct option from the given question and answer choices. You are given a question along with multiple options. Choose the correct option. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",  # Reshad's update
            # "data": load_dataset("LLM-Beetle/arc_translated")["train"],
            # "data": load_dataset("Emirrv/arc_translated_cutted_with_extra_cols_and_options_structure_last_version_mmlu_arc")["train"],
            "data": load_dataset("Emirrv/arc_translated_cutted_with_new_aze_option_structure_latest_mmlu_arc")["train"], # simplified, aze version (question, answer, options column )
        },

        {
            "task_type": "mmlu",
            "dstype": "mmc",
            "group": "math",
            "subtext": "You are an AI designed to solve mathematical word problems in Azerbaijani. Your task is to analyze the given question and select the correct option from the provided choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text.",
            "data": load_dataset("Emirrv/gsm8k_mmc")["train"],
        },

        # {
        #     "task_type": "mmlu_context",
        #     "group": "mmlu",
            # "dstype": "qmc",
        #     "data": load_dataset("LLM-Beetle/anl_quad")["train"],
        # },
        # {
        #     # "task_type": "mmlu_context_check", ??
        #     "group": "context",
            # "dstype": "",
        #     "data": load_dataset("LLM-Beetle/fact_checker")["train"]
        # },

        {
            "task_type": "qa",
            "dstype": "qa",
            "group": "economics",
            "subtext": "",
            "data": load_dataset("LLM-Beetle/LLM_generated_qa_latest")["train"]
        },
        {
            "task_type": "rag",
            "dstype": "cqa",
            "group": "context",
            "subtext": "",
            "data": load_dataset("LLM-Beetle/Quad_benchmark_cqa_latest")["train"]
        },
    ]







    result = {
        "config": {
            # "num_fewshot": num_fewshot,
            # "batch_size": batch_size,
            # "device": device,
            # "limit": limit,
        },
        # "dataset_scores": {},
        # "group_scores": {},
        "results": {},

    }


    print("datasets", datasets)


    for dataset in datasets:
        task_type = dataset['task_type']
        data = dataset['data']
        dstype = dataset['dstype']
        group = dataset['group']

        total_score = 0
        limit = 2

        print("dataset:", dataset)
        print("dstype:", dstype)
        print("tasktype:", task_type)

        # limit = limit or 0  # Sets `limit` to 0 if it's None

        if data:
            total_limit = min(limit, len(data))
        else:
            raise ValueError('errror')

        print("total_limit", total_limit)

        dataset_score = 0
        correct_count = 0


        # base_prompt_mmlu = dynamic_multiple_choice_base_prompt(dataset=data, few_shot=5)
        # base_prompt_subtype_mmlu = dynamic_multiple_choice_subtype_base_prompt(
            # dataset=data, few_shot=5
        # )

        for i in tqdm(range(total_limit), desc=f'Evaluating task_type: {task_type} - dstype: {dstype}'):
            # question = data['question'][i] if data['question'][i] else None
            # if question == None and dstype == 'arc':
            #     question = data['Azerbaijani_q'][i]

            # correct_answer = data['answer'][i]

            # options = data['options'][i] if task_type == "mmlu" and data['options'][i] else None
            # if options == None:
            #     options = data['choices'][i]

            # context = data['context'][i] if task_type in ["rag", "qa"] else None

            # Retrieve the question column dynamically based on the dataset and dstype
            if 'question' in data.column_names:
                question = data['question'][i]
            # elif dstype == 'arc' and 'Azerbaijani_q' in data.column_names:
            #     question = data['Azerbaijani_q'][i]
            # else:
            #     raise KeyError("Neither 'question' nor 'Azerbaijani_q' found in the dataset.")

            # Retrieve the correct answer
            if 'answer' in data.column_names:
                correct_answer = data['answer'][i]
            # elif dstype == 'arc' and 'answerKey' in data.column_names:
            #     correct_answer = data['answerKey'][i]


            # Dynamically select the options column based on what exists
            if 'options' in data.column_names:
                options = data['options'][i]
            elif 'choices' in data.column_names:
                options = data['choices'][i]
            elif task_type in ['qa', 'rag']:
                pass
            else:
                raise KeyError("Neither 'options' nor 'choices' found in the dataset.")

            # Handle the context column if applicable
            context = data['context'][i] if task_type in ["rag"] else None




            if task_type == "mmlu":
                if API:
                    predicted_answer = get_answer_multiple_choice_w_dstype(
                        question=question, options=options, model=model, num_fewshot=0, dstype=dstype, api=True, gguf=False
                    )
                elif GGUF:
                    predicted_answer = get_answer_multiple_choice_w_dstype(
                        question=question, options=options, model=model, num_fewshot=0, dstype=dstype, api=False, gguf=True, repo_id=repo_id
                    )
                else:
                    predicted_answer = get_answer_multiple_choice_w_dstype(
                        question=question, options=options, model=model, num_fewshot=0, dstype=dstype, tokenizer=tokenizer, api=False, gguf=False
                    )
            elif task_type == "qa":
                if API:
                    predicted_answer = get_answer_qa(question, model, api=True, gguf=False)  # Removed tokenizer
                elif GGUF:
                    predicted_answer = get_answer_qa(question, model, repo_id=repo_id, api=False, gguf=True)  # Removed tokenizer
                else:
                    predicted_answer = get_answer_qa(question, model, tokenizer, api=False, gguf=False)
            elif task_type == "rag":
                if API:
                    predicted_answer = get_answer_rag(question, context, model, api=True, gguf=False)  # Removed tokenizer
                elif GGUF:
                    predicted_answer = get_answer_rag(question, context, model, repo_id=repo_id, api=False, gguf=True)  # Removed tokenizer
                else:
                    predicted_answer = get_answer_rag(question, context, model, tokenizer, api=False, gguf=False)
            else:
                raise ValueError("Invalid task type")



            if task_type in ["mmlu", "mmlu_context"]:
                score = compare_answers(correct_answer, predicted_answer)
                print("\n\nscore MMLU:", score, "\n")
                total_score += score
                dataset_score += score
                print("\ntotal_score:", total_score, "\n")
                print("\ndataset_score:", dataset_score, "\n\n")

            elif task_type == "qa":
                score = handle_qa_score(question, correct_answer, predicted_answer) # With GPT eval score
                # score = handle_qa_score(correct_answer, predicted_answer) # Without GPT based eval score
                print("\n\nscore QA:", score, "\n")
                total_score += score
                dataset_score += score
                print("\ntotal_score:", total_score, "\n")
                print("\ndataset_score:", dataset_score, "\n\n")

            elif task_type == "rag":
                score = handle_context_qa_score(question, correct_answer, predicted_answer)
                # score = handle_context_qa_score(correct_answer, predicted_answer) # Without GPT based eval score
                print("\n\nscore RAG:", score, "\n")
                total_score += score
                dataset_score += score
                print("\ntotal_score:", total_score, "\n")
                print("\ndataset_score:", dataset_score, "\n\n")



            # SCORES:  
            # Version 1 (base version)
            # if task_type in ["mmlu", "mmlu_context"]:
            #     score = compare_answers(correct_answer, predicted_answer)
            #     print("\n\nscore:", score, "\n")
            #     total_score += score
            #     print("\ntotal_score:", score, "\n\n")

            # elif task_type == "qa":
            #     score = handle_qa_score(actual_answer=correct_answer, predicted_answer=predicted_answer)
            #     # score = handle_qa_score(question=question, actual_answer=correct_answer, predicted_answer=predicted_answer)
            #     total_score += score

            # elif task_type == "rag":
            #     score = handle_context_qa_score(actual_answer=correct_answer, predicted_answer=predicted_answer)
            #     # score = handle_qa_score(question=question, actual_answer=correct_answer, predicted_answer=predicted_answer)
            #     total_score += score
        
        result["results"][dstype] = {
            "metric_name": total_score / total_limit if total_limit > 0 else 0.0
        }    

        print("\n\n", result["results"], "\n\n")
        # result.save_to_disk('path/to/save/') # save
        # result.to_parquet('path/to/save/dataset.parquet')


# 
    #     average_dataset_score = dataset_score / total_limit if total_limit > 0 else 0.0
    #     result["dataset_scores"][f"{group}_{task_type}"] = average_dataset_score

    #     if group not in result["group_scores"]:
    #         result["group_scores"][group] = {
    #             "total_score": 0,
    #             "total_count": 0
    #         }
    #     result["group_scores"][group]["total_score"] += correct_count
    #     result["group_scores"][group]["total_count"] += total_limit

    # for group, scores in result["group_scores"].items():
    #     total_score = scores["total_score"]
    #     total_count = scores["total_count"]
    #     result["group_scores"][group] = total_score / total_count if total_count > 0 else 0.0



    return result


if __name__ == '__main__':
    # result = evaluate('LLM-Beetle/ProdataAI_Llama-3.1-8bit-Instruct', 5, 32, 'cpu', 10)
    # print(result)

    # print(evaluate('LLM-Beetle/ProdataAI_Llama-3.1-8bit-Instruct',0,0,0,0,0,2,0))
    print(evaluate('**MODEL_NAME**',0,0,0,0,0,2,0))


