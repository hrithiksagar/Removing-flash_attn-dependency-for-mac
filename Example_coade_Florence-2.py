# code to change default GPU on mac which will be CPU to MPS (MPS is a MAC Metal GPU, not CUDA)
import torch

if torch.backends.mps.is_available():
    print("MPS backend is available.")
else:
    print("MPS backend is not available.")
    
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(device, "- Is now assigned to this device")


# code to remove flash_attn dependancy from transformers model in mac metal as it doesnt support metal, only supports NVIDA CUDA
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import os
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

# fix the imports
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

# example use of a Florence 2 model
from transformers import AutoModelForCausalLM, AutoProcessor
model_name = "microsoft/Florence-2-large-ft"
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                             cache_dir="Florence 2",
                                             device_map="mps",
                                             trust_remote_code=True)

