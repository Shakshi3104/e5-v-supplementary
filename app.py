import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlavaNextProcessor, LlavaNextForConditionalGeneration

from PIL import Image


def load_and_resize(name, size) -> Image:
    img = Image.open(name)
    img.thumbnail((size, size))
    return img


if __name__ == '__main__':
    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'

    # model_name ='royokong/e5-v'
    model_name = "models/huggingface/e5-v"

    processor = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to("mps")

    processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

    img_prompt = llama3_template.format('<image>\nSummary above image in one word: ')
    text_prompt = llama3_template.format('<sent>\nSummary above sentence in one word: ')

    max_size = 256
    images = [
        load_and_resize("images/baby-tiger-with-tiramisu.png", max_size),
        load_and_resize("images/strawberry-parfait-emoji.png", max_size),
        load_and_resize("images/christmas-wreath-emoji.png", max_size)
    ]

    texts = [
        "A baby tiger with tiramisu",
        "A strawberry parfait",
        "A christmas wreath"
    ]

    # You may have used the wrong order for inputs. `images` should be passed before `text`.
    img_inputs = processor([img_prompt] * len(images), images, return_tensors="pt", padding=True).to('mps')
    text_inputs = processor([text_prompt.replace('<sent>', text) for text in texts], return_tensors="pt",
                            padding=True).to('mps')

    with torch.no_grad():
        img_embs = model(**img_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
        text_embs = model(**text_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]

        text_embs = F.normalize(text_embs, dim=-1)
        img_embs = F.normalize(img_embs, dim=-1)

    print(text_embs @ img_embs.t())
