
# --- START COPY PASTE FROM FLASK API CODE ---
from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

# Load model
pipe = StableDiffusionPipeline.from_single_file(
    "AnyLoRA_noVae_fp16-pruned.ckpt",
    torch_dtype=torch.float16,
    safety_checker=None,
    use_safetensors=True
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("bitkun-10.safetensors", adapter_name="default")
pipe.set_adapters(["default"], adapter_weights=[0.8])

emotion_tags = {
    "happy": "bitkun, smiling, cartoon mascot style, joyful expression, vibrant colors",
    "sad": "bitkun, teary eyes, gloomy expression, cartoon",
    "angry": "bitkun, red cheeks, intense cartoon, furrowed brows"
    # Add more...
}

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    emotion = data.get("emotion", "happy").lower()
    negative_prompt = data.get("negative_prompt", "")
    num_images = int(data.get("num_images", 1))
    prompt = emotion_tags.get(emotion, emotion_tags["happy"])

    images_base64 = []
    for i in range(num_images):
        torch.manual_seed(42 + i)
        with torch.autocast("cuda"):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                width=512,
                height=512,
                generator=torch.Generator(device="cuda").manual_seed(42 + i)
            )
        image = result.images[0]
        buf = BytesIO()
        image.save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        images_base64.append(img_str)

    return jsonify({"images": images_base64})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
# --- END ---
