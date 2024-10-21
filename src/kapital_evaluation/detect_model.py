"""
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSummarization,
    AutoModelForTranslation,
    AutoModel,
    AutoConfig
)

def load_model_and_tokenizer(model_name, cache_dir='cache': str):
    # Load the model configuration to detect the model type
    config = AutoConfig.from_pretrained(model_name, cache_dir='cache')
    
    # Check the model type from the configuration
    model_type = config.model_type
    
    # Load the tokenizer (same for all models)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='cache')
    
    # Detect model type and load the corresponding model
    if model_type in ['gpt2', 'gpt', 'llama', 'causal-lm']:  
        # Causal Language Model (e.g., GPT, LLAMA)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='cache')
    
    elif model_type in ['bert', 'roberta', 'albert', 'distilbert', 'electra']:
        # Masked Language Models or Sequence Classification (e.g., BERT, RoBERTa)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='cache')
    
    elif model_type in ['t5', 'bart', 'pegasus']:
        # Text-to-Text Models (e.g., T5, BART, PEGASUS for tasks like summarization)
        model = AutoModel.from_pretrained(model_name, cache_dir='cache')
    
    elif model_type in ['xlnet', 'deberta', 'xlm', 'flaubert']:
        # Sequence Classification or Token Classification (e.g., XLNet, DeBERTa)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='cache')
    
    elif model_type in ['roberta', 'electra', 'distilbert', 'albert']:
        # Sequence Classification (e.g., RoBERTa for sentence classification)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='cache')

    elif model_type == 'tfa':
        # Example for other potential models like TFA
        model = AutoModel.from_pretrained(model_name, cache_dir='cache')
    
    elif model_type == 'xlm':
        # For models like XLM, which are multilingual
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='cache')
    
    elif model_type == 'bart':
        # For BART (Summarization, Translation, etc.)
        model = AutoModelForSummarization.from_pretrained(model_name, cache_dir='cache')
    
    elif model_type == 'deberta':
        # DeBERTa (Sequence Classification)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='cache')
    
    # Generic fallback for any other architectures that might not be covered
    else:
        model = AutoModel.from_pretrained(model_name, cache_dir='cache')

    return tokenizer, model

# Example usage
model_name, cache_dir='cache' = "llama-7b"  # Replace with your model name

tokenizer, model = load_model_and_tokenizer(model_name, cache_dir='cache')

print(f"Loaded model: {model_name, cache_dir='cache'}")
print(f"Tokenizer: {tokenizer}")
print(f"Model: {model}")






# --------


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSummarization,
    AutoModelForTranslation,
    AutoModel,
    AutoConfig
)

import torch
from optimum.intel import INCQuantizer  # This is for dynamic quantization via Optimum
from optimum.pipelines import pipeline

def load_and_quantize_model(model_name, cache_dir='cache': str, quantize: bool = False):
    # Load the model configuration to detect the model type
    config = AutoConfig.from_pretrained(model_name, cache_dir='cache')
    
    # Check the model type from the configuration
    model_type = config.model_type
    
    # Load the tokenizer (same for all models)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='cache')
    
    # Load the model based on its type
    if model_type in ['gpt2', 'gpt', 'llama', 'causal-lm']:  
        # Causal Language Model (e.g., GPT, LLAMA)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='cache')
    
    elif model_type in ['bert', 'roberta', 'albert', 'distilbert', 'electra']:
        # Masked Language Models or Sequence Classification (e.g., BERT, RoBERTa)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='cache')
    
    elif model_type in ['t5', 'bart', 'pegasus']:
        # Text-to-Text Models (e.g., T5, BART, PEGASUS for tasks like summarization)
        model = AutoModel.from_pretrained(model_name, cache_dir='cache')
    
    elif model_type in ['xlnet', 'deberta', 'xlm', 'flaubert']:
        # Sequence Classification or Token Classification (e.g., XLNet, DeBERTa)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='cache')
    
    elif model_type == 'bart':
        # For BART (Summarization, Translation, etc.)
        model = AutoModelForSummarization.from_pretrained(model_name, cache_dir='cache')
    
    # Quantize the model if requested
    if quantize:
        model = quantize_model(model)
    
    return tokenizer, model


def quantize_model(model):
    # Using Optimum to dynamically quantize the model to INT8
    # Here we use the Intel-optimized library for model quantization
    quantizer = INCQuantizer.from_pretrained(model)
    quantized_model = quantizer.quantize(model)
    
    # You can also use PyTorch's native dynamic quantization:
    # quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
    return quantized_model


# Example usage
model_name, cache_dir='cache' = "bert-base-uncased"  # Replace with any Hugging Face model
quantized = True  # Set to True to load the quantized model

tokenizer, model = load_and_quantize_model(model_name, cache_dir='cache', quantize=quantized)

print(f"Loaded model: {model_name, cache_dir='cache'}")
print(f"Tokenizer: {tokenizer}")
print(f"Model: {model}")


"""




# --------

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    # AutoModelForSummarization,
    AutoModel,
    AutoConfig
)

import torch

def load_model_and_check_quantization(model_name: str):
    """
    Load the model and check if it is quantized based on parameter types.
    """
    # Load the model configuration to detect the model type
    config = AutoConfig.from_pretrained(model_name, cache_dir='cache')
    
    # Check the model type from the configuration
    model_type = config.model_type
    print("model_type:", model_type)
    
    # Load the tokenizer (same for all models)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='cache')
    
    # Load the model based on its type
    if model_type in ['gpt2', 'gpt', 'llama', 'causal-lm', 'qwen2', 'phi3']:  
        # Causal Language Model (e.g., GPT, LLAMA)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='cache')
    
    elif model_type in ['bert', 'roberta', 'albert', 'distilbert', 'electra']:
        # Masked Language Models or Sequence Classification (e.g., BERT, RoBERTa)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='cache')
    
    elif model_type in ['t5', 'bart', 'pegasus']:
        # Text-to-Text Models (e.g., T5, BART, PEGASUS for tasks like summarization)
        model = AutoModel.from_pretrained(model_name, cache_dir='cache')
    
    elif model_type in ['xlnet', 'deberta', 'xlm', 'flaubert']:
        # Sequence Classification or Token Classification (e.g., XLNet, DeBERTa)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='cache')
    
    else:
        # Generic fallback for any other architectures that might not be covered
        model = AutoModel.from_pretrained(model_name, cache_dir='cache')
    
    # elif model_type == 'bart':
    #     # For BART (Summarization, Translation, etc.)
    #     model = AutoModelForSummarization.from_pretrained(model_name, cache_dir='cache')
    
    # Check if the model is quantized
    # if is_quantized_model(model):
    #     print(f"Model {model_name} is quantized.")
    # else:
    #     print(f"Model {model_name} is not quantized.")
    
    return tokenizer, model


def is_quantized_model(model):
    """
    Check if the model is quantized by inspecting the parameter types.
    Quantized models typically have data types like torch.qint8, torch.float16, etc.
    """
    # List of all common quantized types (can be extended to other formats)
    quantized_types = [
        torch.qint8,  # 8-bit signed integer (INT8)
        torch.uint8,  # 8-bit unsigned integer (UINT8)
        torch.float16,  # 16-bit floating point (FP16)
        torch.bfloat16,  # 16-bit bfloat (BF16)
        # torch.qint4,  # 4-bit signed integer (INT4) (if supported)
        torch.int16,  # 16-bit integer (INT16)
        torch.float32,  # 32-bit floating point (not quantized, but could be part of a hybrid model)
    ]
    
    # Iterate over model parameters and check the dtype
    for name, param in model.named_parameters():
        if param.dtype in quantized_types:
            print(f"Quantized parameter found: {name} with dtype {param.dtype}")
            return True  # The model has quantized weights
    
    return False  # No quantized weights found


def detect_and_print_model_info(model_name: str):
    """
    Detects if the model is quantized and prints relevant information.
    """
    tokenizer, model = load_model_and_check_quantization(model_name)
    
    # Additional information about model
    print(f"Model architecture: {model.config.architectures}")
    print(f"Model's first parameter dtype: {next(model.parameters()).dtype}")
    
    return tokenizer, model




# Example usage
# model_name = "bert-base-uncased"  # Replace with your model name

# # Check if the model is quantized
# tokenizer, model = detect_and_print_model_info(model_name)

# print(f"Loaded model: {model_name}")
# print(f"Tokenizer: {tokenizer}")
# print(f"Model: {model}")

