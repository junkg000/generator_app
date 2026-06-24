import io
import PIL.Image
import concurrent.futures
import google.generativeai as genai

def generate_auto_prompt(model, keywords):
    prompt = f"""
You are an expert product photographer. Create a highly professional, hyper-realistic e-commerce background generation prompt for an Inpainting model.
Product keywords: {keywords}
CRITICAL RULES:
- MUST be hyper-realistic, high-end commercial e-commerce photography.
- NO fantasy, NO magical elements.
- Just output the ENGLISH prompt string directly. Nothing else. No quotes.
Example: A premium dark studio setting resting on elegant marble with realistic soft lighting and shadows wrapping around the object.
"""
    try:
        res = model.generate_content(prompt)
        return res.text.strip()
    except Exception:
        return "A premium dark studio setting with realistic soft lighting and shadows wrapping around the object."

def generate_auto_copy(model, keywords, index):
    if index == 0:
        prompt = f"""
You are a top-tier copywriter. Write a compelling main Title and Description for this product.
Product keywords: {keywords}
CRITICAL: The output MUST be entirely in natural, engaging Korean language (한국어).
Format:
TITLE: [Catchy Main Title in Korean]
DESC: [1-2 sentences of engaging description in Korean]
"""
    else:
        prompt = f"""
You are a top-tier copywriter. Write a compelling section Title and Description for feature #{index} of this product.
Product keywords: {keywords}
CRITICAL: The output MUST be entirely in natural, engaging Korean language (한국어).
Format:
TITLE: [Catchy Section Title in Korean]
DESC: [1-2 sentences of engaging description in Korean]
"""
    try:
        res = model.generate_content(prompt)
        text = res.text.strip()
        lines = text.split('\n')
        title = lines[0].replace('TITLE:', '').strip() if len(lines) > 0 else "Product Feature"
        desc = lines[1].replace('DESC:', '').strip() if len(lines) > 1 else "Amazing product description."
        return title, desc
    except Exception:
        return "Awesome Product", "This is an amazing product."

def process_single_image(idx, file_bytes, keywords, api_key, photoroom_key, replicate_key, engine, advanced_remove_bg, generate_photoroom_bg, generate_replicate_bg):
    # 1. Remove BG
    fg_bytes = advanced_remove_bg(file_bytes, photoroom_key=photoroom_key)
    fg_img = PIL.Image.open(io.BytesIO(fg_bytes)).convert("RGBA")
    
    # Setup Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # 2. Generate AI BG Prompt
    bg_prompt = generate_auto_prompt(model, keywords)
    
    # 3. Generate AI BG
    ai_b64 = None
    ai_mime = "image/png"
    img_data = None
    
    if engine.startswith("Photoroom") and photoroom_key:
        img_data = generate_photoroom_bg(bg_prompt, fg_img, photoroom_key)
    if (img_data is None or isinstance(img_data, dict)) and (engine.startswith("Replicate") or engine.startswith("Photoroom")) and replicate_key:
        img_data = generate_replicate_bg(bg_prompt, fg_img, replicate_key)
    if img_data is None or isinstance(img_data, dict):
        try:
            res = model.generate_content(bg_prompt)
            img_data = res.candidates[0].content.parts[0].inline_data.data
        except Exception:
            pass
            
    import base64
    if img_data and not isinstance(img_data, dict):
        ai_b64 = base64.b64encode(img_data).decode('utf-8')
    
    # 4. Generate Copywriting
    title, desc = generate_auto_copy(model, keywords, idx)
    
    # 5. Return assembled data
    original_b64 = base64.b64encode(file_bytes).decode('utf-8')
    
    return {
        "index": idx,
        "original_b64": original_b64,
        "ai_b64": ai_b64,
        "ai_mime": ai_mime,
        "title": title,
        "desc": desc
    }

def run_autopilot_parallel(files, keywords, api_key, photoroom_key, replicate_key, engine, advanced_remove_bg, generate_photoroom_bg, generate_replicate_bg):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for i, f in enumerate(files):
            file_bytes = f.getvalue()
            futures.append(
                executor.submit(
                    process_single_image, i, file_bytes, keywords, api_key, photoroom_key, replicate_key, engine,
                    advanced_remove_bg, generate_photoroom_bg, generate_replicate_bg
                )
            )
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    results.sort(key=lambda x: x["index"])
    return results
