import modal
from fastapi import UploadFile, File, Form

# ==== Modal Image: All deps installed INSIDE the Modal image ====
custom_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "fastapi[standard]",
        "uvicorn",
        "transformers @ git+https://github.com/huggingface/transformers",
        "accelerate",
        "qwen-vl-utils[decord]==0.0.8",
        "torch",
        "torchvision",
        "torchaudio",
        "pydantic",
    )
)

# ==== Modal App ====
app = modal.App("qwen2_5_vl-inference")

HF_TOKEN = modal.Secret.from_name("hf_token")  # Replace with your Hugging Face token secret name
# HF_TOKEN = "" # Uncomment for local testing without Modal

# ==== Model/Processor Persistent Objects ====
@app.function(image=custom_image, gpu="T4", timeout=60)
@modal.fastapi_endpoint(method="POST")
async def infer(
    image: UploadFile = File(...),
    prompt: str = Form(...)
    ):
    import os
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    # Read uploaded image from POST
    # form = await request.form()
    # upload: UploadFile = form["image"]
    temp_path = f"/tmp/input_img_{os.getpid()}.jpg"
    with open(temp_path, "wb") as f:
        f.write(await image.read())

    # Model/Processor init (singleton per container)
    if not hasattr(infer, "model"):
        infer.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN,
        )
        infer.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", token=HF_TOKEN
        )

    model = infer.model
    processor = infer.processor

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{temp_path}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Clean up uploaded file
    try:
        os.remove(temp_path)
    except Exception:
        pass

    return {"result": output_text[0]}