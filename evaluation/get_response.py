
import argparse
import json

import requests
from PIL import Image
from io import BytesIO
import argparse, json, os, torch
from PIL import Image

from tqdm import tqdm

import datetime
import torch
import requests

from io import BytesIO
from transformers import TextStreamer
import sys
sys.path.append("/home/zhengzhixiao/SeVa/seva")
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def load_image(path: str):
    """Always load from local disk (no internet)."""
    return Image.open(path).convert("RGB")

def main(args):
    disable_torch_init()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 1. build model ----------
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=("llava_lora_model" if args.model_base else model_name),
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=device,
    )
    model = model.to(device)
    model.eval()

    # ---------- 2. load questions ----------
    with open(args.input, "r") as f:
        records = json.load(f)

    results = []
    conv_template = conv_templates["llava_v1"]

    for item in tqdm(records, total=len(records)):
        # (a) resolve image path
        filename = os.path.split(item["image_src"])[1]
        image_path = os.path.join(args.image_root, filename)
        image = load_image(image_path)

        # (b) build prompt
        conv = conv_template.copy()
        conv.append_message(conv.roles[0], None)   # placeholder for image+question
        conv.append_message(conv.roles[1], None)   # placeholder for answer

        prompt = item["question"]
        if model.config.mm_use_im_start_end:
            prompt = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{prompt}"
        else:
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
        conv.messages[0][1] = prompt
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        # (c) tokenize & encode
        input_ids = tokenizer_image_token(
            conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        image_tensor = process_images([image], image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        # (d) generation
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0.0,    
                max_new_tokens=512,
                stopping_criteria=[KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)],
            )
        answer = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        item["model_answer"] = answer
        results.append(item)

    # ---------- 3. save ----------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} answers to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="MMHal response_template.json")
    parser.add_argument("--output", required=True, help="where to save answers")
    parser.add_argument("--image-root", required=True, help="directory with all images")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    main(args)
