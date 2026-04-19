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
    setattr(module,"saved_input",input)
    if output is None:
        return input
    return output

for name, module in model.named_modules():
    module.register_forward_hook(hook)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
with httpx.stream("GET", url) as response:
    image = Image.open(BytesIO(response.read()))

inputs = processor(images=image, return_tensors="pt")

print(inputs["pixel_values"].size())

model(**inputs,return_dict=False)

for name, module in model.named_modules():
    print(name,type(module),)
    for key in ["saved_output","saved_input"]:
        print(f"\t{key}",end=" ")
        out=getattr(module,key,None)
        if out !=None:
            if type(out)==tuple:
                for n in range(len(out)):
                    print(type(out[n]),end=" ")
                    if type(out[n])==torch.Tensor:
                        print((out[n].size()),end=" ")
            elif type(out)==torch.Tensor:
                print(out.size(),end=" ")
            else:
                print(type(out),end=" ")
        print(" ")
