import os
import uuid
import base64
from pathlib import Path
from typing import Optional

import aiofiles
import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Portrait Studio")

BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
GENERATED_DIR = BASE_DIR / "generated"

UPLOAD_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(exist_ok=True)

STYLES = {
    "realistic": {
        "name": "Photorealistic",
        "description": "Ultra-detailed, lifelike photography",
        "emoji": "📷",
        "prompt_suffix": "photorealistic, 8K resolution, professional photography, natural lighting, detailed skin texture, bokeh background",
    },
    "oil_painting": {
        "name": "Oil Painting",
        "description": "Classic fine art with rich brushwork",
        "emoji": "🖼️",
        "prompt_suffix": "oil painting, impasto technique, rich textured brushstrokes, museum quality, old masters style, dramatic chiaroscuro lighting",
    },
    "watercolor": {
        "name": "Watercolor",
        "description": "Soft, translucent washes of color",
        "emoji": "🎨",
        "prompt_suffix": "watercolor painting, soft flowing washes, transparent layers, delicate paper texture, wet on wet technique, luminous colors",
    },
    "anime": {
        "name": "Anime",
        "description": "Japanese animation style",
        "emoji": "⛩️",
        "prompt_suffix": "anime art style, Studio Ghibli inspired, detailed cel shading, vibrant colors, expressive eyes, clean linework",
    },
    "digital_art": {
        "name": "Digital Art",
        "description": "Modern digital illustration",
        "emoji": "💻",
        "prompt_suffix": "digital art, concept art, ArtStation trending, detailed digital painting, professional illustration, vibrant colors",
    },
    "pencil_sketch": {
        "name": "Pencil Sketch",
        "description": "Hand-drawn graphite artwork",
        "emoji": "✏️",
        "prompt_suffix": "pencil sketch, graphite drawing, cross-hatching technique, detailed linework, shading gradients, sketchbook style",
    },
    "pop_art": {
        "name": "Pop Art",
        "description": "Bold Warhol-inspired graphics",
        "emoji": "🔴",
        "prompt_suffix": "pop art style, Andy Warhol inspired, bold flat colors, halftone dots, high contrast, graphic design aesthetic",
    },
    "fantasy": {
        "name": "Fantasy",
        "description": "Magical and mystical artwork",
        "emoji": "🧙",
        "prompt_suffix": "fantasy art style, magical atmosphere, ethereal lighting, mystical elements, epic fantasy illustration, detailed environment",
    },
    "cinematic": {
        "name": "Cinematic",
        "description": "Movie-quality dramatic scenes",
        "emoji": "🎬",
        "prompt_suffix": "cinematic photography, movie still, dramatic lighting, film grain, anamorphic lens flare, color graded, epic composition",
    },
    "vintage": {
        "name": "Vintage",
        "description": "Retro and nostalgic aesthetic",
        "emoji": "📻",
        "prompt_suffix": "vintage photography, 1970s aesthetic, film grain, faded colors, warm tones, retro filter, nostalgic atmosphere",
    },
    "cyberpunk": {
        "name": "Cyberpunk",
        "description": "Neon-lit futuristic dystopia",
        "emoji": "🤖",
        "prompt_suffix": "cyberpunk art style, neon lights, futuristic city, rain-slicked streets, holographic displays, Blade Runner aesthetic, purple and cyan tones",
    },
    "impressionist": {
        "name": "Impressionist",
        "description": "Monet-style light and color",
        "emoji": "🌸",
        "prompt_suffix": "impressionist painting, Monet style, loose visible brushstrokes, captured light and movement, vibrant dabs of color, plein air feeling",
    },
}


class GenerateRequest(BaseModel):
    appearance_description: str
    style: str
    scene_prompt: str
    additional_details: Optional[str] = ""
    image_size: Optional[str] = "1024x1024"


@app.get("/api/styles")
async def get_styles():
    return STYLES


@app.post("/api/analyze")
async def analyze_photo(photo: UploadFile = File(...)):
    if not photo.content_type or not photo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await photo.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be under 10MB")

    photo_id = str(uuid.uuid4())
    ext = Path(photo.filename or "photo.jpg").suffix or ".jpg"
    save_path = UPLOAD_DIR / f"{photo_id}{ext}"
    async with aiofiles.open(save_path, "wb") as f:
        await f.write(contents)

    b64_image = base64.standard_b64encode(contents).decode("utf-8")
    media_type = photo.content_type or "image/jpeg"

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analyze this person's appearance and write a detailed, flowing description "
                            "of approximately 120 words for use in AI image generation. Cover: face shape, "
                            "eye color and shape, hair color/style/length, skin tone, nose shape, lip fullness, "
                            "estimated age range, and any distinctive features (freckles, dimples, facial hair, etc.). "
                            "Write in third person, present tense. Be specific and descriptive. "
                            "Output ONLY the description — no preamble, no labels, no commentary."
                        ),
                    },
                ],
            }
        ],
    )

    appearance_description = ""
    for block in message.content:
        if block.type == "text":
            appearance_description = block.text.strip()
            break

    return {"photo_id": photo_id, "appearance_description": appearance_description}


@app.post("/api/generate")
async def generate_image(request: GenerateRequest):
    if request.style not in STYLES:
        raise HTTPException(status_code=400, detail=f"Unknown style: {request.style}")

    valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
    image_size = request.image_size if request.image_size in valid_sizes else "1024x1024"

    style_info = STYLES[request.style]

    claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    prompt_msg = claude_client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Create an optimized DALL-E 3 image generation prompt. Combine these inputs:\n\n"
                    f"PERSON'S APPEARANCE: {request.appearance_description}\n\n"
                    f"SCENE/SETTING: {request.scene_prompt}\n\n"
                    f"ART STYLE: {style_info['name']} — {style_info['prompt_suffix']}\n\n"
                    f"ADDITIONAL DETAILS: {request.additional_details or 'None'}\n\n"
                    "Write a single, detailed prompt (150-200 words) that seamlessly integrates all elements. "
                    "Emphasize the person's specific physical features so they are clearly identifiable. "
                    "Make the scene vivid and the style explicit. "
                    "Output ONLY the prompt text — no preamble, no labels."
                ),
            }
        ],
    )

    enhanced_prompt = ""
    for block in prompt_msg.content:
        if block.type == "text":
            enhanced_prompt = block.text.strip()
            break

    openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    dalle_response = await openai_client.images.generate(
        model="dall-e-3",
        prompt=enhanced_prompt,
        size=image_size,
        quality="hd",
        style="vivid",
        n=1,
    )

    image_data = dalle_response.data[0]
    image_url = image_data.url
    revised_prompt = getattr(image_data, "revised_prompt", enhanced_prompt)

    image_id = str(uuid.uuid4())
    image_path = GENERATED_DIR / f"{image_id}.png"

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        img_response = await http_client.get(image_url)
        img_response.raise_for_status()

    async with aiofiles.open(image_path, "wb") as f:
        await f.write(img_response.content)

    return {
        "image_id": image_id,
        "image_url": f"/generated/{image_id}.png",
        "style": request.style,
        "style_name": style_info["name"],
        "style_emoji": style_info["emoji"],
        "enhanced_prompt": enhanced_prompt,
        "revised_prompt": revised_prompt,
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "anthropic_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
    }


app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")
