import base64
import json
import os
from pathlib import Path
from openai import OpenAI
from PIL import Image
import io
import config
import traceback

def encode_image_to_base64(image_path: Path) -> str:
    """
    Encodes an image file to a base64 string.
    Resizes if too large to avoid timeouts/errors.
    """
    with Image.open(image_path) as img:
        # Resize if max dimension > 2000 (safe limit for most VL models)
        max_size = 2000
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size))
        
        # Save to buffer as JPEG (lighter than PNG)
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def call_qwen_vl(image_path: Path, prompt: str) -> str:
    """
    Calls the Qwen-VL model with an image and a prompt.
    """
    if not config.QWEN_API_KEY:
         raise ValueError("QWEN_API_KEY is not set. Please check your .env file.")

    client = OpenAI(
        api_key=config.QWEN_API_KEY,
        base_url=config.QWEN_BASE_URL,
        timeout=180.0, # Increased timeout for heavy models
    )

    try:
        base64_image = encode_image_to_base64(image_path)
        
        response = client.chat.completions.create(
            model=config.QWEN_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}" 
                            },
                        },
                    ],
                }
            ],
            temperature=0.1, 
            max_tokens=4096
        )
        
        content = response.choices[0].message.content
        # Debug print to see what model returns
        print(f"\nDEBUG MODEL RESPONSE:\n{content}\n------------------")
        return content

    except Exception as e:
        print(f"Error calling Qwen-VL API for {image_path.name}:")
        traceback.print_exc()
        raise e

def call_llm_text(prompt: str, system_message: str = "Ты полезный ассистент.") -> str:
    client = OpenAI(
        api_key=config.QWEN_API_KEY,
        base_url=config.QWEN_BASE_URL,
    )
    try:
        response = client.chat.completions.create(
            model=config.QWEN_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM API (Text): {e}")
        return ""

def parse_json_from_response(response_text: str):
    """
    Parses JSON from the model's response.
    Robust to markdown, raw JSON, or surrounding text.
    """
    text = response_text.strip()
    
    # Если модель вернула ```json ... ```
    if "```json" in text:
        try:
            text = text.split("```json")[1].split("```")[0].strip()
        except IndexError:
            pass
    elif "```" in text:
        try:
            text = text.split("```")[1].split("```")[0].strip()
        except IndexError:
            pass
    
    # Попытка распарсить как есть
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Если не вышло, возможно там лишний текст.
        # Ищем границы JSON (от первого [ или { до последнего ] или })
        try:
            # Для списка (наш случай в Layout Analysis)
            start_list = text.find("[")
            end_list = text.rfind("]")
            
            # Для объекта
            start_obj = text.find("{")
            end_obj = text.rfind("}")
            
            # Определяем, что идет раньше (список или объект)
            if start_list != -1 and (start_obj == -1 or start_list < start_obj):
                if end_list != -1:
                    candidate = text[start_list : end_list+1]
                    return json.loads(candidate)
            elif start_obj != -1:
                if end_obj != -1:
                    candidate = text[start_obj : end_obj+1]
                    return json.loads(candidate)
                
        except Exception as e:
            print(f"JSON parsing failed even with heuristics: {e}")
            pass
            
        print(f"FAILED TO PARSE JSON. Raw response:\n{response_text}")
        raise
