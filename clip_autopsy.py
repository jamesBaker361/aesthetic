import transformers

from transformers import AutoTokenizer, CLIPTextModel

from PIL import Image
import httpx
from io import BytesIO
from transformers import AutoProcessor, CLIPVisionModel
import torch

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

def hook(module, input, output):
    setattr(module,"saved_output",output)
    if output is None:
        return input
    return output

for name, module in model.named_modules():
    module.register_forward_hook(hook)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
with httpx.stream("GET", url) as response:
    image = Image.open(BytesIO(response.read()))

inputs = processor(images=image, return_tensors="pt")

model(**inputs,return_dict=False)

for name, module in model.named_modules():
    print(name,type(module),end="")
    out=getattr(module,"saved_output",None)
    if out !=None:
        if type(out)==tuple:
            print(out[0].size())
        elif type(out)==torch.Tensor:
            print(out.size())
        else:
            print(type(out))
