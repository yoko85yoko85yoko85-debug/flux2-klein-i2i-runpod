import base64
import io
import threading
import warnings

import runpod
import torch
from PIL import Image, ImageOps
from diffusers import Flux2KleinPipeline

MODEL = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
SIZES = {(640, 640), (768, 768), (1024, 1024), (1024, 768), (768, 1024)}
Image.MAX_IMAGE_PIXELS = 20_000_000
pipe = Flux2KleinPipeline.from_pretrained(
    MODEL, revision=REVISION, torch_dtype=torch.bfloat16, use_safetensors=True,
)
pipe.enable_model_cpu_offload()
lock = threading.Lock()


def parse_request(p):
    if p.get("mode") != "flux2_klein_i2i":
        raise ValueError("FLUX.2 klein用アプリから送信してください。")
    encoded = p.get("image_base64")
    if not isinstance(encoded, str) or not 1 <= len(encoded) <= 4_000_000:
        raise ValueError("元画像が必要です。送信画像は3MB以下にしてください。")
    raw = base64.b64decode(encoded, validate=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("JPEG・PNG・WebPを選んでください。")
            image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = int(p.get("width", 1024)), int(p.get("height", 1024))
    if (width, height) not in SIZES:
        raise ValueError("未対応の出力サイズです。")
    prompt = str(p.get("prompt", "")).strip()
    if not 1 <= len(prompt) <= 3000:
        raise ValueError("編集指示は1〜3000文字で入力してください。")
    seed = int(p.get("seed", 42))
    if not 0 <= seed <= 2147483647:
        raise ValueError("シード値を確認してください。")
    if p.get("lora") is not None or p.get("mask_base64"):
        raise ValueError("この初期版はLoRA・マスク入力には対応していません。")
    # Keep the complete reference image; all dimensions are multiples of 32.
    image = ImageOps.pad(image, (width, height), method=Image.Resampling.LANCZOS, color="black")
    return image, prompt, width, height, seed


def handler(job):
    try:
        image, prompt, width, height, seed = parse_request(job.get("input", {}))
    except Exception as error:
        return {"error": str(error)}
    with lock, torch.inference_mode():
        result = pipe(
            image=image, prompt=prompt, width=width, height=height,
            num_inference_steps=4, guidance_scale=1.0,
            generator=torch.Generator("cpu").manual_seed(seed), num_images_per_prompt=1,
            max_sequence_length=512,
        ).images[0]
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return {
        "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "mime_type": "image/png", "mode": "flux2_klein_i2i", "model": MODEL,
        "width": result.width, "height": result.height, "seed": seed,
        "steps": 4, "guidance": 1.0, "revision": REVISION,
    }


runpod.serverless.start({"handler": handler})
