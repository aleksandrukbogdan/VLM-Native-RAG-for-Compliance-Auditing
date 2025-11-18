import base64
import json
import os
from pathlib import Path
from openai import OpenAI
import config

def encode_image_to_base64(image_path: Path) -> str:
    """
    Encodes an image file to a base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_qwen_vl(image_path: Path, prompt: str) -> str:
    """
    Calls the Qwen-VL model with an image and a prompt.
    
    Args:
        image_path (Path): Path to the image file.
        prompt (str): The text prompt for the model.
        
    Returns:
        str: The text response from the model.
    """
    
    if not config.QWEN_API_KEY:
         raise ValueError("QWEN_API_KEY is not set. Please check your .env file.")

    client = OpenAI(
        api_key=config.QWEN_API_KEY,
        base_url=config.QWEN_BASE_URL,
    )

    base64_image = encode_image_to_base64(image_path)

    try:
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
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.1, # Low temperature for more deterministic output
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Qwen-VL API: {e}")
        raise e

def call_llm_text(prompt: str, system_message: str = "Ты полезный ассистент.") -> str:
    """
    Calls the LLM (Qwen) with text only, no images.
    """
    if not config.QWEN_API_KEY:
         raise ValueError("QWEN_API_KEY is not set. Please check your .env file.")

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
    Parses JSON from the model's response, handling potential markdown formatting.
    """
    try:
        # Remove markdown code blocks if present
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
             cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
            
        return json.loads(cleaned_text.strip())
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from response: {response_text}")
        raise e

