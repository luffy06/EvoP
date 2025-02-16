import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoConfig

def get_llm(model_name, device_map="auto"):

    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype=torch.bfloat16, 
                                                 low_cpu_mem_usage=True,
                                                 device_map=device_map,
                                                 )
    model.seqlen = config.max_position_embeddings
    model.name = model_name

    return model