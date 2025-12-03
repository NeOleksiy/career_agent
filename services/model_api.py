import asyncio
import aiohttp
import tenacity
import json
import os

DEBUG = os.getenv("DEBUG_LLM", "0") == "1"

class ModelAPIError(Exception):
    pass

async def get_completion(url, token, messages, model, temperature=0.7, **kwargs):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    
    formatted_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        if role not in ["system", "user", "assistant"]:
            role = "user"
        
        content = msg.get("content") or msg.get("text") or ""
        if content:
            formatted_messages.append({
                "role": role,
                "content": content
            })
    
    payload = {
        "model": model,
        "messages": formatted_messages,
        "temperature": temperature,
        "max_tokens": 1024,
        "stream": False
    }
    
    timeout = aiohttp.ClientTimeout(total=60)
    
    if DEBUG:
        print("[LLM][Mistral][REQ]", json.dumps(payload, ensure_ascii=False, indent=2))
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            url=url, 
            json=payload, 
            headers=headers
        ) as resp:
            text = await resp.text()
            
            if DEBUG:
                print(f"[LLM][Mistral][HTTP] {resp.status}")
                print(f"[LLM][Mistral][RAW] {text[:2000]}")
            
            if resp.status != 200:
                raise ModelAPIError(f"Mistral HTTP {resp.status}: {text}")
            
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                raise ModelAPIError(f"Invalid JSON from Mistral: {text[:500]}")
            
            if DEBUG:
                print("[LLM][Mistral][PARSED]", data)
            
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"].strip()
                    return content
                elif "message" in choice and "text" in choice["message"]:
                    content = choice["message"]["text"].strip()
                    return content
            
            if "outputs" in data and len(data["outputs"]) > 0:
                content = data["outputs"][0].get("text", "").strip()
                return content
            
            raise ModelAPIError(f"Unexpected Mistral response format: {json.dumps(data, ensure_ascii=False)[:1000]}")

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
    retry=tenacity.retry_if_exception_type((aiohttp.ClientError, ModelAPIError))
)
async def wrapped_get_completion(*args, **kwargs):
    try:
        return await get_completion(*args, **kwargs)
    except Exception as e:
        error_msg = f"[Error Mistral]: {str(e)}"
        if DEBUG:
            print(error_msg)
        return f'{{"response": "", "error": "{str(e)}"}}'