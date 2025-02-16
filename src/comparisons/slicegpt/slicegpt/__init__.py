# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys
for i, sys_path in enumerate(sys.path):
    if sys_path == os.path.dirname(__file__):
        sys.path[i] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from comparisons.slicegpt.slicegpt.adapters.llama_adapter import LlamaModelAdapter
from comparisons.slicegpt.slicegpt.adapters.opt_adapter import OPTModelAdapter
from comparisons.slicegpt.slicegpt.adapters.phi2_adapter import Phi2ModelAdapter
from comparisons.slicegpt.slicegpt.adapters.phi3_adapter import Phi3ModelAdapter
from comparisons.slicegpt.slicegpt.data_utils import get_dataset, prepare_dataloader
from comparisons.slicegpt.slicegpt.gpu_utils import benchmark, distribute_model, evaluate_ppl
from comparisons.slicegpt.slicegpt.hf_utils import get_model_and_tokenizer, load_sliced_model
from comparisons.slicegpt.slicegpt.layernorm_fusion import fuse_modules, replace_layers
from comparisons.slicegpt.slicegpt.model_adapter import LayerAdapter, ModelAdapter
from comparisons.slicegpt.slicegpt.rotate import rotate_and_slice

__all__ = ["data_utils", "gpu_utils", "hf_utils", "layernorm_fusion", "rotate"]
