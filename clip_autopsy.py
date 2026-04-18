import transformers

from transformers import AutoTokenizer, CLIPTextModel

model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")


def hook(module, input, output):
    setattr(module,"saved_output",output)
    if output is None:
        return input
    return output

for name, module in model.named_modules():
    module.register_forward_hook(hook)

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled (EOS token) states

for name, module in model.named_modules():
    print(name,type(module),getattr(module,"saved_output").size())
