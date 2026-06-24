import streamlit as st
import base64
import os
import io
import json
import datetime
import PIL.Image
from duckduckgo_search import DDGS
import google.generativeai as genai



def prepare_inpaint_inputs(fg_img):
    import PIL.Image
    import io
    import numpy as np
    
    bg_size = max(fg_img.size)
    bg_size = (bg_size, bg_size) # Make it square for better stable diffusion results
    
    # Base image: Transparent bg replaced with white
    base_img = PIL.Image.new("RGB", bg_size, (255, 255, 255))
    offset = ((bg_size[0] - fg_img.size[0]) // 2, (bg_size[1] - fg_img.size[1]) // 2)
    base_img.paste(fg_img, offset, fg_img)
    
    # Mask image: white for background (inpaint), black for foreground (keep)
    fg_padded = PIL.Image.new("RGBA", bg_size, (0, 0, 0, 0))
    fg_padded.paste(fg_img, offset, fg_img)
    alpha = np.array(fg_padded.split()[-1])
    mask_np = np.where(alpha > 0, 0, 255).astype(np.uint8)
    mask_img = PIL.Image.fromarray(mask_np, mode="L")
    
    base_bytes = io.BytesIO()
    base_img.save(base_bytes, format="PNG")
    base_bytes.seek(0)
    
    mask_bytes = io.BytesIO()
    mask_img.save(mask_bytes, format="PNG")
    mask_bytes.seek(0)
    
    return base_bytes, mask_bytes, offset

def generate_photoroom_bg(prompt, fg_img, api_key):
    import requests
    import io
    
    try:
        url = 'https://image-api.photoroom.com/v2/edit'
        headers = {'x-api-key': api_key.strip(), 'pr-ai-background-model-version': 'background-studio-beta-2025-03-17'}
        
        img_bytes = io.BytesIO()
        fg_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {'image_file': img_bytes}
        data = {'background.prompt': prompt}
        
        res = requests.post(url, headers=headers, files=files, data=data)
        if res.status_code == 200:
            return res.content
        else:
            print(f"Photoroom BG Error: {res.status_code} - {res.text}")
            return {"error": f"{res.status_code} - {res.text}"}
    except Exception as e:
        print("Photoroom BG Exception:", e)
        return {"error": str(e)}

def generate_replicate_bg(prompt, fg_img, replicate_key):
    import replicate
    import io
    import PIL.Image
    
    try:
        base_bytes, mask_bytes, offset = prepare_inpaint_inputs(fg_img)
        
        client = replicate.Client(api_token=replicate_key)
        
        output = client.run(
            "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
            input={
                "prompt": prompt,
                "image": base_bytes,
                "mask": mask_bytes,
                "num_outputs": 1,
                "num_inference_steps": 25,
                "guidance_scale": 7.5
            }
        )
        
        if output and len(output) > 0:
            import requests
            img_data = requests.get(output[0]).content
            return img_data
        return None
    except Exception as e:
        print("Replicate Error:", e)
        return {"error": str(e)}

def generate_contextual_prompts(fg_img, api_key, use_replicate=False):
    import google.generativeai as genai
    import json
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
        
        if use_replicate:
            prompt = '''
You are a professional product photography art director. Analyze the provided image.
Provide exactly 4 high-quality image generation prompts for an Inpainting model to generate a realistic background AND human interaction.
The object in the image will be PERFECTLY PRESERVED. Your prompt must describe the ENTIRE scene seamlessly blending with the object.
The prompts must be in English.
CRITICAL RULES: 
- MUST be hyper-realistic, high-end commercial e-commerce photography.
- NO fantasy, NO magical elements, NO glowing auras, NO floating objects. 
- Must look like a real photo taken with a DSLR camera (e.g., 85mm lens, beautiful bokeh, studio lighting or natural sunlight).

Requirements for the 4 backgrounds:
1. "studio_1": A premium dark studio setting (e.g., dark slate or black marble) with realistic soft lighting and shadows wrapping around the object. 
2. "studio_2": A beautiful human model naturally WEARING or HOLDING the object. Describe the model's skin texture, realistic wrist/hand/neck, and clean lighting. MUST look like a real fashion photoshoot.
3. "lifestyle": A realistic lifestyle setting with bright, warm colors (e.g., beige and gold tones, cozy sunlight, soft fabrics).
4. "creative": A modern, pastel-toned room or clean creative space (e.g., soft pink/blue pastel background, modern props).
CRITICAL: Do NOT say "empty in the center". The object is already there. Describe the person wearing it or the environment interacting with it!
Return ONLY a valid JSON object with the keys "studio_1", "studio_2", "lifestyle", "creative", and the prompt strings as values.
'''
        else:
            prompt = '''
You are a professional product photography art director. Analyze the provided image (which is a subject with its background removed).
Provide exactly 4 high-quality image generation prompts to create a background that perfectly matches this subject.
The prompts must be in English and optimized for an image generation model (like Imagen or Midjourney).
Requirements for the 4 backgrounds:
1. "studio_1": A premium studio setting that perfectly matches the subject's material and vibe (e.g., marble for cosmetics, wood for food).
2. "studio_2": Another studio setting with a different mood (e.g., dark dramatic lighting, or bright pastel).
3. "lifestyle": A realistic lifestyle setting where this object would naturally be placed or used (e.g., a cozy living room table, a sunny window sill, a person's hand holding it).
4. "creative": A highly creative, cinematic, or thematic background (e.g., surrounded by nature, floating on water, cyberpunk, etc.).
Important: The background must be completely empty in the center where the subject will be placed. End each prompt with "completely empty in the center, perfect for product placement, no text, 8k resolution".
Return ONLY a valid JSON object with the keys "studio_1", "studio_2", "lifestyle", "creative", and the prompt strings as values.
'''

        res = model.generate_content([prompt, fg_img])
        data = json.loads(res.text)
        return [
            data.get("studio_1", "A high-end minimalist studio background, soft lighting, completely empty in the center, 8k resolution"),
            data.get("studio_2", "A bright and airy studio background, marble surface, completely empty in the center, 8k resolution"),
            data.get("lifestyle", "A realistic lifestyle setting, natural lighting, out of focus, completely empty in the center, 8k resolution"),
            data.get("creative", "A cinematic thematic background, creative lighting, completely empty in the center, 8k resolution")
        ]
    except Exception as e:
        print("Prompt generation failed:", e)
        # Fallback to hardcoded prompts
        return [
            "A very elegant minimalist dark studio background for product photography, dramatic soft spotlight in the center, 8k resolution, completely empty, no text.",
            "A bright and airy minimalist studio background with soft natural morning light and gentle shadows, pure white marble surface, 8k resolution, completely empty, no text.",
            "A luxurious warm gold and beige studio background, soft bokeh, high-end product photography style, completely empty, 8k resolution, no text.",
            "A modern abstract geometric background in pastel tones, soft lighting, 3d render style, completely empty, perfect for product placement, no text."
        ]

def global_create_fallback_bg(theme_idx, size):
    import PIL.Image
    from PIL import ImageDraw
    bg = PIL.Image.new("RGBA", size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(bg)
    if theme_idx == 0:
        for y in range(size[1]):
            ratio = y / size[1]
            r, g, b = int(58 - 45 * ratio), int(62 - 48 * ratio), int(71 - 54 * ratio)
            draw.line([(0, y), (size[0], y)], fill=(r, g, b, 255))
    elif theme_idx == 1:
        for y in range(size[1]):
            ratio = y / size[1]
            r, g, b = int(255 - 20 * ratio), int(255 - 20 * ratio), int(255 - 15 * ratio)
            draw.line([(0, y), (size[0], y)], fill=(r, g, b, 255))
    elif theme_idx == 2:
        for y in range(size[1]):
            ratio = y / size[1]
            r, g, b = int(212 - 50 * ratio), int(175 - 40 * ratio), int(55 - 10 * ratio)
            draw.line([(0, y), (size[0], y)], fill=(r, g, b, 255))
    else:
        for y in range(size[1]):
            ratio = y / size[1]
            r, g, b = int(245 - 20 * ratio), int(240 - 20 * ratio), int(250 - 20 * ratio)
            draw.line([(0, y), (size[0], y)], fill=(r, g, b, 255))
    return bg



st.set_page_config(page_title="프리미엄 상세페이지 생성기", layout="wide")

@st.cache_resource
def get_rembg_session():
    from rembg import new_session
    return new_session('birefnet-general')

def advanced_remove_bg(img_bytes, use_matting=False, model_idx=1, photoroom_key=""):
    if photoroom_key and photoroom_key.strip():
        try:
            import requests
            url = 'https://sdk.photoroom.com/v1/segment'
            headers = {'x-api-key': photoroom_key.strip()}
            files = {'image_file': img_bytes}
            res = requests.post(url, headers=headers, files=files)
            if res.status_code == 200:
                return res.content
            else:
                st.toast(f'Photoroom API 오류: {res.text}. 기본 AI로 대체합니다.')
        except Exception as e:
            st.toast(f'Photoroom 연동 오류: {e}. 기본 AI로 대체합니다.')
            
    # Fallback to rembg
    from rembg import remove, new_session
    if model_idx == 1:
        session = new_session('u2net')
        return remove(img_bytes, session=session, post_process_mask=False)
    elif model_idx == 2:
        session = new_session('birefnet-general')
        return remove(img_bytes, session=session, post_process_mask=False)
    elif model_idx == 3:
        session = new_session('isnet-general-use')
        return remove(img_bytes, session=session, post_process_mask=False)
    else:
        session = get_rembg_session()
        return remove(img_bytes, session=session, post_process_mask=True, alpha_matting=use_matting)

def remove_small_noise(img):
    try:
        import cv2
        import numpy as np
        arr = np.array(img)
        alpha = arr[:, :, 3]
        _, binary_alpha = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_alpha, connectivity=8)
        if num_labels <= 1:
            return img
        
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = np.max(areas)
        threshold = max_area * 0.05 # 가장 큰 물체 크기의 5% 미만인 파편은 모두 제거
        
        new_alpha = np.zeros_like(alpha)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= threshold:
                new_alpha[labels == i] = alpha[labels == i]
                
        arr[:, :, 3] = new_alpha
        return PIL.Image.fromarray(arr)
    except Exception as e:
        print(f"Noise removal error: {e}")
        return img

SAVE_DIR = os.path.join(os.getcwd(), "saved_products")
os.makedirs(SAVE_DIR, exist_ok=True)

def load_product(folder_name):
    path = os.path.join(SAVE_DIR, folder_name, "data.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            st.session_state.loaded_name = data.get("name", "")
            st.session_state.loaded_desc = data.get("description", "")
            st.session_state.loaded_search_kw = data.get("search_keyword", "")
            
            st.session_state.kw_brand = data.get("brand_name", "")
            st.session_state.kw_modifier = data.get("modifier", "")
            st.session_state.kw_main = data.get("main_keyword", data.get("name", ""))
            st.session_state.kw_sub1 = data.get("sub_keyword1", "")
            st.session_state.kw_sub2 = data.get("sub_keyword2", "")
            st.session_state.loaded_cs_opt = data.get("cs_option", "옵션 A: 해송 (010-4506-0728)")
            st.session_state.loaded_hero_b64 = data.get("hero_b64", "")
            st.session_state.loaded_mime_hero = data.get("mime_hero", "")
            
            # 하위 호환성 및 신규 스토리 블록 로딩
            story_blocks = data.get("story_blocks", [])
            old_grids = data.get("grid_b64s", [])
            old_mimes = data.get("mime_grids", [])
            
            if not story_blocks and old_grids:
                for b64, mime in zip(old_grids, old_mimes):
                    story_blocks.append({"b64": b64, "mime": mime, "text": ""})
            
            st.session_state.loaded_story_blocks = story_blocks
            st.session_state.auto_desc = data.get("description", "")
            st.session_state.load_timestamp = str(datetime.datetime.now().timestamp())
            
def optimize_image(uploaded_file, max_width=860, quality=80):
    uploaded_file.seek(0)
    img = PIL.Image.open(uploaded_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w, h = img.size
    if w > max_width:
        img = img.resize((max_width, int(h * max_width / w)), PIL.Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=quality)
    return base64.b64encode(buffered.getvalue()).decode(), "image/jpeg"

def save_product(name, desc, search_kw, cs_opt, hero_b64, mime_hero, story_blocks, brand_name="", modifier="", main_keyword="", sub_keyword1="", sub_keyword2=""):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c.isspace()]).rstrip()
    if not safe_name: safe_name = "product"
    folder_name = f"{safe_name}_{timestamp}"
    folder_path = os.path.join(SAVE_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    data = {
        "name": name,
        "description": desc,
        "search_keyword": search_kw,
        "brand_name": brand_name,
        "modifier": modifier,
        "main_keyword": main_keyword,
        "sub_keyword1": sub_keyword1,
        "sub_keyword2": sub_keyword2,
        "cs_option": cs_opt,
        "hero_b64": hero_b64,
        "mime_hero": mime_hero,
        "story_blocks": story_blocks
    }
    with open(os.path.join(folder_path, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return folder_name

st.title("🎨 럭셔리 스토리텔링 상세페이지 생성기")
st.markdown("사진 사이사이에 나만의 이야기를 적어 홈쇼핑/백화점 퀄리티의 명품 상세페이지를 완성하세요.")

# ===========================
# 사이드바
# ===========================
st.sidebar.header("⚙️ AI 설정")
st.sidebar.markdown("[무료 Gemini API 발급받기](https://aistudio.google.com/app/apikey)")

api_key_file = "api_key.txt"
saved_key = ""
if os.path.exists(api_key_file):
    with open(api_key_file, "r") as f:
        saved_key = f.read().strip()

api_key = st.sidebar.text_input("Gemini API Key 입력", type="password", value=saved_key)
col_k1, col_k2 = st.sidebar.columns(2)
with col_k1:
    if st.button("내 컴퓨터에 영구 저장", use_container_width=True):
        if api_key:
            with open(api_key_file, "w") as f:
                f.write(api_key)
            st.sidebar.success("저장 완료!")
with col_k2:
    if st.button("저장된 키 삭제", use_container_width=True):
        if os.path.exists(api_key_file):
            os.remove(api_key_file)
        st.sidebar.success("삭제 완료!")
        import time
        time.sleep(0.5)
        st.rerun()

pr_key_path = "photoroom_key.txt"
saved_pr_key = ""
if os.path.exists(pr_key_path):
    with open(pr_key_path, "r", encoding="utf-8") as f:
        saved_pr_key = f.read().strip()

photoroom_api_key = st.sidebar.text_input("✨ Photoroom API 키 (선택사항)", type="password", value=saved_pr_key, help="포토룸 API 키를 입력하면 기존 무료 AI 대신 포토룸의 초고화질 누끼 AI를 사용합니다.")
col_pk1, col_pk2 = st.sidebar.columns(2)
with col_pk1:
    if st.button("키 저장", key="save_pr"):
        with open(pr_key_path, "w", encoding="utf-8") as f:
            f.write(photoroom_api_key)
        st.sidebar.success("저장 완료!")
with col_pk2:
    if st.button("키 삭제", key="del_pr"):
        if os.path.exists(pr_key_path):
            os.remove(pr_key_path)
        st.sidebar.success("삭제 완료!")

st.sidebar.markdown("---")
st.sidebar.markdown("🚀 **고급 착용샷 AI (Replicate)**")
st.sidebar.markdown("[👉 Replicate API 토큰 발급](https://replicate.com/)")

rep_key_path = "replicate_key.txt"
saved_rep_key = ""
if os.path.exists(rep_key_path):
    with open(rep_key_path, "r", encoding="utf-8") as f:
        saved_rep_key = f.read().strip()

replicate_api_key = st.sidebar.text_input("🔑 Replicate API 토큰", type="password", value=saved_rep_key, help="r8_ 로 시작하는 토큰을 입력하세요.")
col_rk1, col_rk2 = st.sidebar.columns(2)
with col_rk1:
    if st.button("토큰 저장", key="save_rep"):
        with open(rep_key_path, "w", encoding="utf-8") as f:
            f.write(replicate_api_key)
        st.sidebar.success("저장 완료!")
with col_rk2:
    if st.button("토큰 삭제", key="del_rep"):
        if os.path.exists(rep_key_path):
            os.remove(rep_key_path)
        st.sidebar.success("삭제 완료!")

st.sidebar.markdown("---")
st.sidebar.markdown("⚙️ **AI 배경 합성 엔진 설정**")
ai_engine = st.sidebar.radio(
    "메인 배경 합성 엔진 선택",
    options=["Photoroom API", "Replicate API", "Gemini (무료)"],
    index=0,
    help="Photoroom: 상업용 쇼핑몰 특화 (추천)\nReplicate: 자유도 높은 인페인팅\nGemini: 무료 대체 엔진"
)
st.session_state.ai_engine = ai_engine

st.sidebar.markdown("---")
st.sidebar.header("📂 내 작업 (자동 저장)")
saved_folders = sorted(os.listdir(SAVE_DIR), reverse=True) if os.path.exists(SAVE_DIR) else []
if not saved_folders:
    st.sidebar.info("저장된 제품이 없습니다.")
else:
    for folder in saved_folders:
        if st.sidebar.button(f"📂 {folder}", key=folder):
            load_product(folder)
            st.rerun()
            
if st.sidebar.button("✨ 새 작업 만들기 (초기화)"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ===========================
# 🚀 오토파일럿 (원클릭 자동 완성)
# ===========================
st.markdown("---")
with st.expander("🚀 [원클릭 자동 완성] 사진 여러 장으로 전체 페이지 한 번에 만들기!", expanded=True):
    st.markdown("**1. 제품 사진들을 한 번에 올려주세요. (최대 6장)**")
    auto_files = st.file_uploader("다중 파일 업로드", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'], key="auto_files")
    
    st.markdown("**2. 제품의 핵심 키워드를 입력해 주세요.**")
    auto_keywords = st.text_input("예: 노란색 구슬 팔찌, 고급스러움, 우아한, 여성용", key="auto_keywords")
    
    if st.button("✨ 1초 만에 상세페이지 자동 완성하기", use_container_width=True, type="primary"):
        if not api_key:
            st.error("좌측 사이드바에 Gemini API 키를 먼저 입력해 주세요!")
        elif not auto_files:
            st.error("사진을 최소 1장 이상 업로드해 주세요!")
        elif not auto_keywords:
            st.error("제품 키워드를 입력해 주세요!")
        else:
            with st.spinner("AI가 피사체를 분리하고, 최고의 배경을 합성하며, 광고 문구를 작성 중입니다... (약 10~30초 소요)"):
                import autopilot
                
                # Limit to 6 files (1 hero + 5 stories)
                process_files = auto_files[:6]
                
                results = autopilot.run_autopilot_parallel(
                    process_files, auto_keywords, api_key, photoroom_api_key, replicate_api_key, st.session_state.get('ai_engine', 'Photoroom API'),
                    advanced_remove_bg, generate_photoroom_bg, generate_replicate_bg
                )
                
                for res in results:
                    idx = res["index"]
                    if idx == 0:
                        # Hero
                        st.session_state.loaded_hero_b64 = res["original_b64"]
                        st.session_state.hero_ai_b64 = res["ai_b64"]
                        st.session_state.hero_ai_mime = res["ai_mime"]
                        st.session_state.hero_fg_img = PIL.Image.new("RGBA", (1,1)) # Dummy to avoid errors
                        st.session_state.kw_main = auto_keywords
                        st.session_state.kw_modifier = res["title"]
                    else:
                        # Story
                        story_idx = idx - 1
                        if 'loaded_story_blocks' not in st.session_state:
                            st.session_state.loaded_story_blocks = []
                        while len(st.session_state.loaded_story_blocks) <= story_idx:
                            st.session_state.loaded_story_blocks.append({})
                        st.session_state.loaded_story_blocks[story_idx]['b64'] = res["original_b64"]
                        st.session_state.loaded_story_blocks[story_idx]['mime'] = res["ai_mime"]
                        st.session_state.loaded_story_blocks[story_idx]['text'] = f"{res['title']}\n{res['desc']}"
                        
                        if 'story_ai_blocks' not in st.session_state:
                            st.session_state.story_ai_blocks = [None] * 5
                        st.session_state.story_ai_blocks[story_idx] = {'b64': res["ai_b64"], 'mime': res["ai_mime"]}
                
                st.success("🎉 상세페이지 초안이 완벽하게 생성되었습니다! 아래에서 디테일하게 수정해 보세요.")
                import time
                time.sleep(2)
                st.rerun()

st.markdown("---")
# ===========================
# 1. 메인 타이틀 & 대표 이미지
# ===========================
st.header("1. 메인 타이틀 & 대표 이미지")
st.markdown("**상세페이지 최상단에 배치될 대표 얼굴입니다.**")

def on_hero_change():
    st.session_state.pop('hero_ai_b64', None)
    st.session_state.pop('hero_ai_mime', None)

def synced_slider(label, min_val, max_val, default_val, step, key_prefix):
    val_key = f"{key_prefix}_val"
    if val_key not in st.session_state:
        st.session_state[val_key] = default_val

    def on_slider():
        v = st.session_state[f"{key_prefix}_slider"]
        st.session_state[val_key] = v
        st.session_state[f"{key_prefix}_num"] = v
    def on_num():
        v = st.session_state[f"{key_prefix}_num"]
        st.session_state[val_key] = v
        st.session_state[f"{key_prefix}_slider"] = v

    col1, col2 = st.columns([3, 1])
    with col1:
        st.slider(label, min_val, max_val, st.session_state[val_key], step, key=f"{key_prefix}_slider", on_change=on_slider)
    with col2:
        st.number_input(label, min_val, max_val, st.session_state[val_key], step, key=f"{key_prefix}_num", on_change=on_num, label_visibility="collapsed")
    return st.session_state[val_key]

@st.dialog("🎨 세부 편집기 (팝업창)", width="large")
def render_editor(target_id="hero"):
    # 이전 편집 상태(슬라이더, 텍스트 등) 복구
    pers = st.session_state.get(f"persistent_editor_{target_id}", {})
    for k, v in pers.items():
        if k not in st.session_state:
            if "rembg" not in k and "reset" not in k:
                st.session_state[k] = v

    if f'{target_id}_fg_img' in st.session_state and f'{target_id}_bg_img' in st.session_state:
        st.markdown("**[조작 방법]** 슬라이더나 텍스트를 변경하면 아래 결과 이미지가 실시간으로 업데이트됩니다.")
        
        # 레이아웃 2분할 (좌: 미리보기, 우: 컨트롤러)
        main_col1, main_col2 = st.columns([1, 1.2])
        
        with main_col1:
            # 미리보기 이미지를 그릴 공간
            preview_container = st.empty()
            
        with main_col2:
            tab1, tab2, tab3, tab4 = st.tabs(["🎯 피사체 조절", "🖼️ 배경 변경", "✍️ 글씨 오버레이", "✨ 포스터 템플릿"])
            
            with tab1:
                rotation = synced_slider("회전 (각도)", -180, 180, 0, 1, f'edit_rotation_{target_id}')
                scale = synced_slider("피사체 크기 조절 (배율)", 0.1, 2.0, 1.0, 0.05, f'edit_scale_{target_id}')
                offset_x = synced_slider("가로 위치 (X)", -1000, 1000, 0, 10, f'edit_x_{target_id}')
                offset_y = synced_slider("세로 위치 (Y)", -1000, 1000, 0, 10, f'edit_y_{target_id}')
                fill_blur_bg = st.toggle("✨ 빈 공간을 원본 사진 블러로 채우기", value=False, key=f'edit_fill_blur_{target_id}')
                overlay_fg = st.toggle("✨ 원본 피사체 덮어쓰기 (Replicate 착용샷 시 끄기)", value=True, key=f'edit_overlay_{target_id}', help="AI 합성 손이나 객체를 가리지 않으려면 끄세요.")
                erode_size = synced_slider("테두리 색번짐 제거 (픽셀 깎기)", 0, 10, 0, 1, f'edit_erode_{target_id}')
                
                st.markdown("---")
                st.markdown("**그림자 세밀 조절**")
                shadow_intensity = synced_slider("그림자 진하기 (투명도)", 0, 255, 180, 10, f'edit_shadow_int_{target_id}')
                shadow_blur = synced_slider("그림자 퍼짐 정도 (크기/블러)", 0, 10, 1, 1, f'edit_shadow_blur_{target_id}')
                shadow_offset_y = synced_slider("그림자 상하 위치 조정", -200, 200, 0, 5, f'edit_shadow_y_{target_id}')
                
                st.markdown("---")
                st.markdown("**✨ 원본 이미지 배경 제거 (누끼따기)**")
                rembg_model_label = st.selectbox(
                    "사용할 AI 엔진 선택 (결과물이 어색할 때 변경해보세요)",
                    [
                        "1. 기본 부드러운 모델 (u2net)",
                        "2. 최신 고성능 모델 (birefnet-general)",
                        "3. 초정밀 외곽선 모델 (isnet-general-use)",
                        "4. 테두리 스무딩 적용 (u2net + post process)"
                    ],
                    key=f'edit_rembg_model_{target_id}'
                )
                
                if st.button("✂️ 선택한 엔진으로 누끼따기", key=f"edit_rembg_{target_id}", help="AI가 사진 속 피사체만 남기고 배경을 투명하게 지워줍니다."):
                    with st.spinner("AI가 배경을 제거하는 중... 잠시만 기다려주세요!"):
                        import io
                        import PIL.Image
                        img_byte_arr = io.BytesIO()
                        st.session_state[f'{target_id}_fg_img'].save(img_byte_arr, format='PNG')
                        
                        m_idx = 0
                        if rembg_model_label.startswith("1"): m_idx = 1
                        elif rembg_model_label.startswith("2"): m_idx = 2
                        elif rembg_model_label.startswith("3"): m_idx = 3
                        
                        fg_bytes = advanced_remove_bg(img_byte_arr.getvalue(), use_matting=False, model_idx=m_idx, photoroom_key=photoroom_api_key)
                        out = PIL.Image.open(io.BytesIO(fg_bytes)).convert("RGBA")
                            
                        st.session_state[f'{target_id}_fg_img'] = out
                        # 렌더링 캐시 초기화
                        st.session_state.pop(f'{target_id}_last_fg_params', None)
            with tab2:
                bg_type = st.radio("배경 설정 방식", ["AI 배경 변경", "단색 배경 적용", "직접 이미지 업로드"], horizontal=True, key=f"edit_bg_type_{target_id}")
                
                bg_upload = None
                use_solid_bg = False
                ai_bg_index = 0
                
                if bg_type == "직접 이미지 업로드":
                    bg_upload = st.file_uploader("배경 사진 직접 업로드", type=['png','jpg','jpeg'], key=f'edit_bg_file_{target_id}')
                elif bg_type == "단색 배경 적용":
                    use_solid_bg = True
                    bg_color = st.color_picker("단색 배경 색상 선택", "#000000", key=f'edit_bg_color_{target_id}')
                else:
                    bg_color = "#000000"
                    
                    if st.button("✨ AI 배경 새로 생성하기", key=f"edit_gen_ai_{target_id}"):
                        saved_key = ""
                        pass # removed import os to fix UnboundLocalError
                        if os.path.exists("api_key.txt"):
                            with open("api_key.txt", "r") as f:
                                saved_key = f.read().strip()
                        if not saved_key:
                            st.warning("사이드바에 Gemini API 키를 먼저 입력해주세요.")
                        else:
                            with st.spinner("Stable Diffusion 인페인팅으로 완벽한 착용샷/배경을 합성 중입니다... (약 15~20초 소요)"):
                                try:
                                    import google.generativeai as genai
                                    import io
                                    import concurrent.futures
                                    genai.configure(api_key=saved_key)
                                    model = genai.GenerativeModel('models/gemini-2.5-flash-image')
                                    fg_for_prompt = st.session_state[f'{target_id}_fg_img']
                                    
                                    pass # removed import os to fix UnboundLocalError
                                    rep_key = ""
                                    if os.path.exists("replicate_key.txt"):
                                        with open("replicate_key.txt", "r", encoding="utf-8") as f:
                                            rep_key = f.read().strip()
                                            
                                    bg_prompts = generate_contextual_prompts(fg_for_prompt, saved_key, use_replicate=bool(rep_key))
                                    generated_bgs = []
                                    def generate_single_bg(prompt):
                                        pass # removed import os to fix UnboundLocalError
                                        engine = st.session_state.get('ai_engine', 'Photoroom API')
                                        img_data = None
                                        
                                        if engine == "Photoroom API" or engine.startswith("Photoroom"):
                                            pr_key = ""
                                            if os.path.exists("photoroom_key.txt"):
                                                with open("photoroom_key.txt", "r", encoding="utf-8") as f: pr_key = f.read().strip()
                                            if pr_key:
                                                img_data = generate_photoroom_bg(prompt, fg_for_prompt, pr_key)
                                                if img_data and isinstance(img_data, dict):
                                                    st.session_state[f'{target_id}_replicate_error'] = img_data.get("error", "Unknown error")
                                                    img_data = None
                                        
                                        if (img_data is None) and (engine == "Replicate API" or engine.startswith("Replicate") or engine.startswith("Photoroom")):
                                            rep_key = ""
                                            if os.path.exists("replicate_key.txt"):
                                                with open("replicate_key.txt", "r", encoding="utf-8") as f: rep_key = f.read().strip()
                                            if rep_key:
                                                img_data = generate_replicate_bg(prompt, fg_for_prompt, rep_key)
                                                if img_data and isinstance(img_data, dict):
                                                    st.session_state[f'{target_id}_replicate_error'] = img_data.get("error", "Unknown error")
                                                    img_data = None
                                                    
                                        if img_data is None:
                                            try:
                                                res = model.generate_content(prompt)
                                                img_data = res.candidates[0].content.parts[0].inline_data.data
                                            except Exception:
                                                img_data = None
                                        return img_data
                                                
                                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                                        results = list(executor.map(generate_single_bg, bg_prompts))
                                    for b_data in results:
                                        if b_data:
                                            generated_bgs.append(b_data)
                                    
                                    while len(generated_bgs) < 4:
                                        idx = len(generated_bgs)
                                        # Use target_id's fg_img size for fallback
                                        target_size = st.session_state[f'{target_id}_fg_img'].size
                                        fallback = global_create_fallback_bg(idx, target_size)
                                        out_bytes = io.BytesIO()
                                        fallback.save(out_bytes, format='PNG')
                                        generated_bgs.append(out_bytes.getvalue())
                                        
                                    st.session_state[f'{target_id}_ai_bg_candidates'] = generated_bgs
                                    
                                    if f'{target_id}_replicate_error' in st.session_state:
                                        err = st.session_state[f'{target_id}_replicate_error']
                                        st.error(f"⚠️ Replicate API 결제 잔액 부족 또는 오류로 인해 무료 제미나이(Gemini)로 대체 생성되었습니다.\n\n(상세: {err})\n\n완벽한 인페인팅 착용샷을 원하시면 Replicate 크레딧을 충전해주세요!")
                                        del st.session_state[f'{target_id}_replicate_error']
                                    else:
                                        st.success("배경 4종이 성공적으로 생성되었습니다! 아래에서 선택해주세요.")
                                except Exception as e:
                                    st.error(f"AI 배경 생성 실패: {e}")

                    if f'{target_id}_ai_bg_candidates' in st.session_state and len(st.session_state[f'{target_id}_ai_bg_candidates']) > 0:
                        theme_names = ["1. 다크 스튜디오", "2. 리얼 모델 착용샷", "3. 포근한 라이프스타일", "4. 모던 파스텔 룸"]
                        opts = theme_names[:len(st.session_state[f'{target_id}_ai_bg_candidates'])]
                        selected_theme = st.selectbox("✨ 4가지 AI 배경 중 선택", opts, key=f'edit_ai_bg_idx_{target_id}')
                        ai_bg_index = opts.index(selected_theme)
                
            with tab3:
                overlay_text = st.text_input("삽입할 문구", "", key=f'edit_text_{target_id}')
                
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    font_family = st.selectbox("글꼴", ["나눔고딕", "나눔명조", "검은고딕 (매우 굵음)"], key=f'edit_font_{target_id}')
                with col_f2:
                    if font_family == "검은고딕 (매우 굵음)":
                        font_weight = st.selectbox("굵기", ["보통"], key=f'edit_weight_{target_id}')
                    else:
                        font_weight = st.selectbox("굵기", ["보통", "굵게"], key=f'edit_weight_{target_id}')
                        
                text_size = synced_slider("글씨 크기", 10, 300, 80, 5, f'edit_text_size_{target_id}')
                text_x = synced_slider("글씨 가로 위치 (좌우 이동)", -1000, 1000, 0, 10, f'edit_text_x_{target_id}')
                text_y = synced_slider("글씨 세로 위치 (상하 이동)", 0, 1500, 100, 10, f'edit_text_y_{target_id}')
                text_color = st.color_picker("글씨 색상", "#FFFFFF", key=f'edit_color_{target_id}')
                
            with tab4:
                st.markdown("**포스터 자동 완성 템플릿**")
                use_template = st.toggle("템플릿 적용하기", value=False, key=f"tmpl_enable_{target_id}")
                
                tmpl_styles = [
                    "1. 클래식 하단 배너", "2. 모던 중앙 집중형", "3. 좌측 세로 리본", "4. 우측 하단 미니멀 박스",
                    "5. 투컬럼 스플릿", "6. 플로팅 포인트 카드", "7. 상하 분리형 (헤더&푸터)",
                    "8. 풀 오버레이 그라데이션", "9. 대각선 스포트라이트", "10. 원형 배지 강조형"
                ]
                selected_tmpl = st.selectbox("적용할 레이아웃 스타일", tmpl_styles, key=f"tmpl_style_{target_id}")
                
                tmpl_color = st.color_picker("배너/포인트 색상", "#4A533E", key=f"tmpl_color_{target_id}")
                tmpl_title = st.text_area("메인 타이틀", "프리미엄\n소나무붓\n4종 세트", key=f"tmpl_title_{target_id}")
                col_t1, col_t2 = st.columns([2, 1])
                with col_t1: tmpl_title_size = synced_slider("타이틀 크기(%)", 50, 300, 100, 5, f"tmpl_title_size_{target_id}")
                with col_t2: tmpl_title_color = st.color_picker("타이틀 색상", "#FFFFFF", key=f"tmpl_title_color_{target_id}")
                
                tmpl_sub_top = st.text_input("상단 서브 타이틀", "한 붓의 차이가 작품의 품격을 만듭니다.", key=f"tmpl_sub_top_{target_id}")
                col_st1, col_st2 = st.columns([2, 1])
                with col_st1: tmpl_sub_top_size = synced_slider("상단 서브 크기(%)", 50, 300, 100, 5, f"tmpl_sub_top_size_{target_id}")
                with col_st2: tmpl_sub_top_color = st.color_picker("상단 서브 색상", "#FFFFFF", key=f"tmpl_sub_top_color_{target_id}")
                
                tmpl_sub_bottom = st.text_input("하단 서브 타이틀", "전통의 깊이, 완성의 차이", key=f"tmpl_sub_bottom_{target_id}")
                col_sb1, col_sb2 = st.columns([2, 1])
                with col_sb1: tmpl_sub_bottom_size = synced_slider("하단 서브 크기(%)", 50, 300, 100, 5, f"tmpl_sub_bottom_size_{target_id}")
                with col_sb2: tmpl_sub_bottom_color = st.color_picker("하단 서브 색상", "#FFFFFF", key=f"tmpl_sub_bottom_color_{target_id}")
                
                st.markdown("---")
                st.markdown("**포인트 텍스트 설정**")
                tmpl_show_icons = st.toggle("포인트 아이콘 표시", value=True, key=f"tmpl_show_icons_{target_id}")
                col_pt1, col_pt2 = st.columns([2, 1])
                with col_pt1: tmpl_p_title_size = synced_slider("포인트 제목 크기(%)", 50, 300, 100, 5, f"tmpl_p_title_size_{target_id}")
                with col_pt2: tmpl_p_title_color = st.color_picker("포인트 제목 색상", "#FFFFFF", key=f"tmpl_p_title_color_{target_id}")
                col_pd1, col_pd2 = st.columns([2, 1])
                with col_pd1: tmpl_p_desc_size = synced_slider("포인트 설명 크기(%)", 50, 300, 100, 5, f"tmpl_p_desc_size_{target_id}")
                with col_pd2: tmpl_p_desc_color = st.color_picker("포인트 설명 색상", "#DDDDDD", key=f"tmpl_p_desc_color_{target_id}")
                
                st.markdown("---")
                if st.button("🔄 위치 0으로 모두 초기화", key=f"reset_tmpl_pos_{target_id}"):
                    st.session_state[f"tmpl_shape_off_x_{target_id}_val"] = 0
                    st.session_state[f"tmpl_text_off_x_{target_id}_val"] = 0
                    st.session_state[f"tmpl_shape_off_y_{target_id}_val"] = 0
                    st.session_state[f"tmpl_text_off_y_{target_id}_val"] = 0

                col_off1, col_off2 = st.columns(2)
                with col_off1:
                    tmpl_shape_off_x = synced_slider("도형 가로 이동 (좌우)", -1000, 1000, 0, 10, f"tmpl_shape_off_x_{target_id}")
                    tmpl_text_off_x = synced_slider("글씨 가로 이동 (좌우)", -1000, 1000, 0, 10, f"tmpl_text_off_x_{target_id}")
                with col_off2:
                    tmpl_shape_off_y = synced_slider("도형 세로 이동 (상하)", -1000, 1000, 0, 10, f"tmpl_shape_off_y_{target_id}")
                    tmpl_text_off_y = synced_slider("글씨 세로 이동 (상하)", -1000, 1000, 0, 10, f"tmpl_text_off_y_{target_id}")
                    
                st.markdown("---")
                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    tmpl_p1_icon = st.text_input("포인트 1 아이콘", "🎯", key=f"tmpl_p1_icon_{target_id}")
                    tmpl_p1_title = st.text_input("포인트 1 제목", "정밀도 98%", key=f"tmpl_p1_title_{target_id}")
                    tmpl_p1_desc = st.text_input("포인트 1 설명", "섬세한 표현력", key=f"tmpl_p1_desc_{target_id}")
                with col_t2:
                    tmpl_p2_icon = st.text_input("포인트 2 아이콘", "🖌️", key=f"tmpl_p2_icon_{target_id}")
                    tmpl_p2_title = st.text_input("포인트 2 제목", "100% 수제 제작", key=f"tmpl_p2_title_{target_id}")
                    tmpl_p2_desc = st.text_input("포인트 2 설명", "장인의 손길로 완성", key=f"tmpl_p2_desc_{target_id}")
                with col_t3:
                    tmpl_p3_icon = st.text_input("포인트 3 아이콘", "🌿", key=f"tmpl_p3_icon_{target_id}")
                    tmpl_p3_title = st.text_input("포인트 3 제목", "천연 소나무 축", key=f"tmpl_p3_title_{target_id}")
                    tmpl_p3_desc = st.text_input("포인트 3 설명", "가볍고 균형 잡힌 사용감", key=f"tmpl_p3_desc_{target_id}")
                
            st.markdown("---")
            col_save1, col_save2 = st.columns([1, 1])
            with col_save1:
                save_btn = st.button("✔️ 이대로 적용하기", type="primary", use_container_width=True)
            with col_save2:
                cancel_btn = st.button("❌ 취소 (창 닫기)", use_container_width=True)
            
        # 실시간 재합성 로직
        try:
            import PIL.Image
            from PIL import ImageFilter, ImageDraw, ImageFont
            import io, base64
            
            fg = st.session_state[f'{target_id}_fg_img']
            bg_base = st.session_state[f'{target_id}_bg_img']
            
            # 1. FG 및 그림자 캐싱
            current_fg_params = (erode_size, scale, shadow_intensity, shadow_blur, rotation)
            if st.session_state.get(f'{target_id}_last_fg_params') != current_fg_params:
                temp_fg = fg.copy()
                if erode_size > 0:
                    import cv2
                    import numpy as np
                    arr = np.array(temp_fg)
                    alpha = arr[:, :, 3]
                    kernel = np.ones((erode_size, erode_size), np.uint8)
                    alpha = cv2.erode(alpha, kernel, iterations=1)
                    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
                    arr[:, :, 3] = alpha
                    temp_fg = PIL.Image.fromarray(arr)
                
                if rotation != 0:
                    temp_fg = temp_fg.rotate(-rotation, expand=True, resample=PIL.Image.Resampling.BICUBIC)
                
                new_size = (int(temp_fg.size[0] * scale), int(temp_fg.size[1] * scale))
                if new_size[0] > 0 and new_size[1] > 0:
                    # 미리보기용은 LANCZOS 대신 BILINEAR 사용하여 속도 향상 가능하나, 캐싱되므로 LANCZOS 유지
                    temp_fg = temp_fg.resize(new_size, PIL.Image.Resampling.LANCZOS)
                
                # 그림자 고속 생성 (CV2 GaussianBlur)
                shadow = PIL.Image.new("RGBA", new_size, (0, 0, 0, 0))
                if shadow_intensity > 0:
                    shadow.paste((0, 0, 0, shadow_intensity), (0, 0), mask=temp_fg)
                
                rad = int(max(new_size) * 0.01 * shadow_blur)
                if rad > 0 and shadow_intensity > 0:
                    import cv2
                    import numpy as np
                    arr = np.array(shadow)
                    ksize = rad * 2 + 1
                    arr = cv2.GaussianBlur(arr, (ksize, ksize), 0)
                    shadow = PIL.Image.fromarray(arr)
                
                st.session_state[f'{target_id}_cached_fg'] = temp_fg
                st.session_state[f'{target_id}_cached_shadow'] = shadow
                st.session_state[f'{target_id}_last_fg_params'] = current_fg_params
            
            fg = st.session_state[f'{target_id}_cached_fg']
            shadow = st.session_state[f'{target_id}_cached_shadow']
            new_size = fg.size

            # 2. BG 캐싱
            bg_up_val = bg_upload.getvalue() if bg_upload is not None else None
            current_bg_params = (bg_up_val, use_solid_bg, bg_color, ai_bg_index, fill_blur_bg)
            if st.session_state.get(f'{target_id}_last_bg_params') != current_bg_params:
                if fill_blur_bg:
                    orig_fg = st.session_state[f'{target_id}_fg_img']
                    import cv2
                    import numpy as np
                    arr = np.array(orig_fg.convert("RGB"))
                    blur_bg = cv2.resize(arr, (bg_base.size[0], bg_base.size[1]))
                    ksize = int(max(bg_base.size) * 0.05) * 2 + 1
                    blur_bg = cv2.GaussianBlur(blur_bg, (ksize, ksize), 0)
                    blur_bg = (blur_bg * 0.8).astype(np.uint8)
                    temp_bg = PIL.Image.fromarray(blur_bg).convert("RGBA")
                else:
                    temp_bg = bg_base.copy()
                    if bg_upload is not None:
                        temp_bg = PIL.Image.open(io.BytesIO(bg_up_val)).convert("RGBA")
                        if temp_bg.size != bg_base.size:
                            temp_bg = temp_bg.resize(bg_base.size, PIL.Image.Resampling.LANCZOS)
                    elif use_solid_bg:
                        temp_bg = PIL.Image.new("RGBA", bg_base.size, bg_color)
                    else:
                        if f'{target_id}_ai_bg_candidates' in st.session_state and len(st.session_state[f'{target_id}_ai_bg_candidates']) > ai_bg_index:
                            temp_bg = PIL.Image.open(io.BytesIO(st.session_state[f'{target_id}_ai_bg_candidates'][ai_bg_index])).convert("RGBA")
                            if temp_bg.size != bg_base.size:
                                temp_bg = temp_bg.resize(bg_base.size, PIL.Image.Resampling.LANCZOS)
                
                st.session_state[f'{target_id}_cached_bg'] = temp_bg
                st.session_state[f'{target_id}_last_bg_params'] = current_bg_params

            bg = st.session_state[f'{target_id}_cached_bg'].copy()
            
            cx = (bg.size[0] - new_size[0]) // 2 + offset_x
            cy = (bg.size[1] - new_size[1]) // 2 + offset_y
            shadow_y = cy + int(max(new_size)*0.02) + shadow_offset_y
            
            temp_layer = PIL.Image.new("RGBA", bg.size, (0, 0, 0, 0))
            if shadow_intensity > 0:
                temp_layer.paste(shadow, (cx, shadow_y), mask=shadow)
            if overlay_fg:
                temp_layer.paste(fg, (cx, cy), mask=fg)
            
            bg = PIL.Image.alpha_composite(bg, temp_layer)
            draw = ImageDraw.Draw(bg)
            
            if use_template:
                import templates
                import importlib
                importlib.reload(templates)
                points = [
                    (tmpl_p1_icon, tmpl_p1_title, tmpl_p1_desc),
                    (tmpl_p2_icon, tmpl_p2_title, tmpl_p2_desc),
                    (tmpl_p3_icon, tmpl_p3_title, tmpl_p3_desc)
                ]
                style_idx = tmpl_styles.index(selected_tmpl)
                tmpl_opts = {
                    "title_size": tmpl_title_size / 100.0, "title_color": tmpl_title_color,
                    "sub_top_size": tmpl_sub_top_size / 100.0, "sub_top_color": tmpl_sub_top_color,
                    "sub_bot_size": tmpl_sub_bottom_size / 100.0, "sub_bot_color": tmpl_sub_bottom_color,
                    "p_title_size": tmpl_p_title_size / 100.0, "p_title_color": tmpl_p_title_color,
                    "p_desc_size": tmpl_p_desc_size / 100.0, "p_desc_color": tmpl_p_desc_color,
                    "show_icons": tmpl_show_icons
                }
                bg = templates.render_template(
                    bg, style_idx, tmpl_color, tmpl_title, tmpl_sub_top, tmpl_sub_bottom, points,
                    tmpl_shape_off_x, tmpl_shape_off_y, tmpl_text_off_x, tmpl_text_off_y, tmpl_opts
                )
                draw = ImageDraw.Draw(bg)
                
            elif overlay_text.strip():
                
                font_map = {
                    "나눔고딕": {"보통": "fonts/NanumGothic-Regular.ttf", "굵게": "fonts/NanumGothic-Bold.ttf"},
                    "나눔명조": {"보통": "fonts/NanumMyeongjo-Regular.ttf", "굵게": "fonts/NanumMyeongjo-Bold.ttf"},
                    "검은고딕 (매우 굵음)": {"보통": "fonts/BlackHanSans-Regular.ttf", "굵게": "fonts/BlackHanSans-Regular.ttf"}
                }
                
                font_path = font_map.get(font_family, {}).get(font_weight, "malgun.ttf")
                if not os.path.exists(font_path):
                    font_path = "malgun.ttf"
                    
                try:
                    font = ImageFont.truetype(font_path, text_size)
                except:
                    try:
                        font = ImageFont.truetype("malgun.ttf", text_size)
                    except:
                        try:
                            font = ImageFont.truetype("AppleGothic.ttf", text_size)
                        except:
                            font = ImageFont.load_default()
                
                try:
                    bbox = draw.textbbox((0, 0), overlay_text, font=font)
                    tw = bbox[2] - bbox[0]
                except:
                    tw = text_size * len(overlay_text) * 0.5
                    
                tx = (bg.size[0] - tw) // 2 + text_x
                draw.text((tx, text_y), overlay_text, fill=text_color, font=font)
            
            out_bytes = io.BytesIO()
            bg.convert("RGB").save(out_bytes, format='PNG')
            encoded = base64.b64encode(out_bytes.getvalue()).decode('utf-8')
            
            # 미리보기 공간에 이미지 그리기 (왼쪽 컬럼)
            preview_container.image(bg, use_container_width=False)
            
            # 저장 버튼 동작 처리
            if save_btn:
                # 현재 에디터의 위젯 상태를 영구 보존용 딕셔너리에 저장
                state_dict = {}
                for k in list(st.session_state.keys()):
                    if k.endswith(f"_{target_id}") and (k.startswith("edit_") or k.startswith("tmpl_")):
                        if "rembg" not in k and "reset" not in k:
                            state_dict[k] = st.session_state[k]
                st.session_state[f"persistent_editor_{target_id}"] = state_dict

                if target_id == "hero":
                    st.session_state.hero_ai_b64 = encoded
                    st.session_state.hero_ai_mime = "image/png"
                else:
                    idx = int(target_id.split('_')[1])
                    if 'loaded_story_blocks' not in st.session_state:
                        st.session_state.loaded_story_blocks = []
                    while len(st.session_state.loaded_story_blocks) <= idx:
                        st.session_state.loaded_story_blocks.append({})
                    st.session_state.loaded_story_blocks[idx]['b64'] = encoded
                    st.session_state.loaded_story_blocks[idx]['mime'] = "image/png"
                    
                    if 'story_ai_blocks' not in st.session_state:
                        st.session_state.story_ai_blocks = [None] * 5
                    st.session_state.story_ai_blocks[idx] = {'b64': encoded, 'mime': "image/png"}
                st.rerun()
            if cancel_btn:
                st.rerun()
                    
        except Exception as e:
            st.error(f"편집 오류: {e}")

col1, col2 = st.columns([1, 1])
with col1:
    hero_file = st.file_uploader("🌟 대표 이미지 1장", accept_multiple_files=False, type=['jpg','png','jpeg'], key="hero_file_main", on_change=on_hero_change)
    if st.session_state.get('hero_ai_b64'):
        import base64
        st.image(base64.b64decode(st.session_state.hero_ai_b64), width=450)
        st.success("✨ AI 고퀄리티 사진이 적용되었습니다!")
        if st.button("❌ 원본 사진으로 복구", key="revert_hero"):
            on_hero_change()
            st.rerun()
            
        # [NEW] 에디터 호출 (팝업창)
        if st.button("🎨 세부 편집기 열기 (새창)", key="hero_editor_btn_2"):
            render_editor("hero")
        
    elif hero_file:
        st.image(hero_file, width=450)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            keep_bg = st.checkbox("원본 배경 유지 (누끼 안 땀)", value=True)
        with col_c2:
            use_matting = st.checkbox("테두리 부드럽게 (털/복잡한 선)", value=False, disabled=keep_bg)
            
        if st.button("⚡ 바로 편집하기 (새창)", type="secondary"):
            with st.spinner("이미지 준비 중... ✂️"):
                import io, PIL.Image
                
                if keep_bg:
                    fg = PIL.Image.open(hero_file).convert("RGBA")
                else:
                    img_bytes = hero_file.getvalue()
                    fg_bytes = advanced_remove_bg(img_bytes, use_matting=use_matting, model_idx=0, photoroom_key=photoroom_api_key)
                    fg = PIL.Image.open(io.BytesIO(fg_bytes)).convert("RGBA")
                
                # 기본 배경은 투명으로 설정 (나중에 단색/이미지로 변경 가능)
                bg = PIL.Image.new("RGBA", fg.size, (255, 255, 255, 0))
                
                st.session_state.hero_fg_img = fg
                st.session_state.hero_bg_img = bg
                st.session_state.hero_ai_b64 = None # 초기화
            render_editor("hero")
    elif st.session_state.get('loaded_hero_b64'):
        import base64
        st.image(base64.b64decode(st.session_state.loaded_hero_b64), width=450)
        st.info("✅ AI 생성/보관함 이미지가 대기 중입니다.")
with col2:
    if "kw_brand" not in st.session_state: st.session_state.kw_brand = ""
    if "kw_main" not in st.session_state: st.session_state.kw_main = ""
    if "kw_modifier" not in st.session_state: st.session_state.kw_modifier = ""
    if "kw_sub1" not in st.session_state: st.session_state.kw_sub1 = ""
    if "kw_sub2" not in st.session_state: st.session_state.kw_sub2 = ""

    def auto_fill_keywords():
        if not st.session_state.kw_main.strip():
            st.toast("⚠️ 메인 키워드를 먼저 입력해주세요!")
            return
        if not api_key:
            st.toast("⚠️ 좌측 사이드바에 API 키를 입력해주세요!")
            return
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            brand_ctx = f"'{st.session_state.kw_brand}' (이 브랜드를 반드시 유지)" if st.session_state.kw_brand.strip() else "'해송' (이 브랜드명을 무조건 사용할 것)"
            mod_ctx = f"'{st.session_state.kw_modifier}' (유지)" if st.session_state.kw_modifier.strip() else "매력적인 수식어 추천"
            seo_target = st.session_state.get("seo_source", "네이버 쇼핑")
            prompt = f"""
다음 핵심 상품 키워드 '{st.session_state.kw_main}'를 기반으로 {seo_target} 상품명 최적화 공식을 완성해줘. JSON 형식으로만 응답해. 키는 'brand', 'modifier', 'sub1', 'sub2'야.

[절대 규칙 - 반드시 지킬 것]
1. 무조건 100% 한국어(Korean)로만 작성해. 영어 단어나 알파벳을 절대 섞지 마.
2. 마크다운(** 등)이나 특수문자를 쓰지 말고 순수 텍스트만 적어.
3. 수식어(modifier)는 15자 이내로 짧게 작성해.

[조건]
- 브랜드명: {brand_ctx}
- 수식어: {mod_ctx}
- sub1, sub2: {seo_target} 연관 검색량이 많을 만한 핵심 서브키워드 추천

출력 예시: {{"brand": "해송", "modifier": "초보자용 부드러운", "sub1": "민화붓", "sub2": "캘리그라피"}}
"""
            response = model.generate_content(prompt)
            import json
            res_text = response.text.strip()
            if res_text.startswith("```json"): res_text = res_text[7:]
            if res_text.startswith("```"): res_text = res_text[3:]
            if res_text.endswith("```"): res_text = res_text[:-3]
            parsed = json.loads(res_text.strip())
            
            if not st.session_state.kw_brand.strip(): st.session_state.kw_brand = parsed.get("brand", "해송")
            if not st.session_state.kw_modifier.strip(): st.session_state.kw_modifier = parsed.get("modifier", "")
            if not st.session_state.kw_sub1.strip(): st.session_state.kw_sub1 = parsed.get("sub1", "")
            if not st.session_state.kw_sub2.strip(): st.session_state.kw_sub2 = parsed.get("sub2", "")
        except Exception as e:
            st.toast(f"❌ AI 분석 실패: {e}")

    col_ai1, col_ai2 = st.columns([1, 1])
    with col_ai1:
        st.markdown("**📦 상품 키워드 입력**")
        st.radio("검색어 최적화(SEO) 출처", ["네이버 쇼핑", "쿠팡"], key="seo_source", horizontal=True)
    with col_ai2:
        st.write("") # add padding to align button
        st.button("✨ 빈칸 AI 자동 추천", on_click=auto_fill_keywords, use_container_width=True)

    kw_col1, kw_col2 = st.columns(2)
    with kw_col1:
        brand_name = st.text_input("브랜드명", key="kw_brand", placeholder="예: 해송")
        main_keyword = st.text_input("메인 키워드 (필수)", key="kw_main", placeholder="예: 세필붓")
        sub_keyword1 = st.text_input("서브 키워드 1", key="kw_sub1", placeholder="예: 민화붓")
    with kw_col2:
        modifier = st.text_input("수식어", key="kw_modifier", placeholder="예: 부드러운")
        sub_keyword2 = st.text_input("서브 키워드 2", key="kw_sub2", placeholder="예: 동양화")

    parts = []
    brand_val = brand_name.strip() if brand_name.strip() else "해송"
    parts.append(brand_val)
    if modifier.strip(): parts.append(modifier.strip())
    if main_keyword.strip(): parts.append(main_keyword.strip())
    if sub_keyword1.strip(): parts.append(sub_keyword1.strip())
    if sub_keyword2.strip(): parts.append(sub_keyword2.strip())
    
    product_name = " ".join(parts) if parts else "상품명 미입력"
    st.markdown("**📋 자동 완성 상품명 (우측 상단 복사 버튼 클릭)**")
    st.code(product_name, language="text")
    search_keyword = main_keyword.strip() if main_keyword.strip() else product_name

if hero_file is not None or st.session_state.get('loaded_hero_b64'):
    col_btn1, col_btn2 = st.columns([1, 0.01])
    with col_btn1:
        if st.button("✨ 텍스트 자동 작성 (검색 연동)"):
            if not api_key:
                st.error("👈 좌측 사이드바에 API 키를 입력해주세요!")
            else:
                with st.spinner("AI가 분석 중입니다... 🤖🔍"):
                    try:
                        web_info = ""
                        if search_keyword.strip():
                            try:
                                with DDGS() as ddgs:
                                    results1 = ddgs.text(f"{search_keyword}", max_results=3)
                                    for res in results1:
                                        web_info += f"- [일반정보] {res['title']}: {res['body']}\n"
                                        
                                    results2 = ddgs.text(f"{search_keyword} 네이버 블로그 쇼핑 리뷰", max_results=3)
                                    for res in results2:
                                        web_info += f"- [네이버리뷰] {res['title']}: {res['body']}\n"
                                st.session_state.last_web_info = web_info
                            except Exception:
                                pass
                                
                        genai.configure(api_key=api_key)
                        if hero_file:
                            img = PIL.Image.open(hero_file)
                        else:
                            import base64
                            img = PIL.Image.open(io.BytesIO(base64.b64decode(st.session_state.loaded_hero_b64)))
                        
                        if web_info.strip():
                            prompt = f"상품 메인 사진과 검색정보야.\n[검색 정보]\n{web_info}\n\n[필수 포함 키워드]\n- 브랜드명: {brand_name}\n- 수식어: {modifier}\n- 메인키워드: {main_keyword}\n- 서브키워드: {sub_keyword1}, {sub_keyword2}\n\n위 [검색 정보]가 너무 일반적이더라도, 반드시 위 [필수 포함 키워드]들의 진짜 의미와 목적을 문맥에 맞게 잘 살려서 내용에 자연스럽게 녹여 줘!\n구구절절 긴 문장은 절대 피하고, 전체 글자수 50자 이내로 네이버 스마트스토어에 올리기 딱 좋은 간결하고 시선을 끄는 카피라이팅을 2~3줄로 작성해 줘.\n\n그리고 이 상품의 필수 포함 키워드들에 관련된 '네이버 스마트스토어' 및 '인스타그램' 마케팅용 추천 해시태그 20개를 작성해줘. 단, 한 줄에 모두 적지 말고 10개씩 2줄로 나누어서 작성해 줘.\n출력 형식은 다음과 같이 구분해줘:\n[카피라이팅]\n(내용)\n[해시태그]\n#태그1 #태그2 ... (10개)\n#태그11 #태그12 ... (10개)"
                        else:
                            prompt = f"이 사진의 특징과 디테일을 파악하되, 아래의 [필수 포함 키워드]들의 의미와 목적을 문맥에 맞게 잘 살려서 카피라이팅에 자연스럽게 녹여 줘.\n\n[필수 포함 키워드]\n- 브랜드명: {brand_name}\n- 수식어: {modifier}\n- 메인키워드: {main_keyword}\n- 서브키워드: {sub_keyword1}, {sub_keyword2}\n\n구구절절 긴 문장은 절대 피하고 전체 글자수 50자 이내로 네이버 스마트스토어에 올리기 딱 좋은 간결하고 시선을 끄는 카피라이팅을 2~3줄로 작성해 줘.\n\n그리고 이 상품의 필수 포함 키워드들에 관련된 '네이버 쇼핑' 및 '인스타그램' 마케팅용 추천 해시태그 20개를 작성해줘. 단, 한 줄에 모두 적지 말고 10개씩 2줄로 나누어서 작성해 줘.\n출력 형식은 다음과 같이 구분해줘:\n[카피라이팅]\n(내용)\n[해시태그]\n#태그1 #태그2 ... (10개)\n#태그11 #태그12 ... (10개)"
                        
                        try:
                            model = genai.GenerativeModel('gemini-2.5-flash')
                            response = model.generate_content([prompt, img])
                        except Exception:
                            model = genai.GenerativeModel('gemini-flash-latest')
                            response = model.generate_content([prompt, img])
                            
                        res_text = response.text
                        if "[해시태그]" in res_text:
                            parts = res_text.split("[해시태그]")
                            st.session_state.auto_desc = parts[0].replace("[카피라이팅]", "").strip()
                            st.session_state.auto_tags = parts[1].strip()
                        else:
                            st.session_state.auto_desc = res_text
                            st.session_state.auto_tags = ""
                        
                        if hero_file:
                            import base64
                            bytes_data = hero_file.getvalue()
                            st.session_state.loaded_hero_b64 = base64.b64encode(bytes_data).decode('utf-8')
                            st.session_state.loaded_hero_mime = hero_file.type
                        
                        st.session_state.load_timestamp = str(datetime.datetime.now().timestamp())
                        st.rerun()
                    except Exception as e:
                        st.error(f"오류: {e}")
desc_default = st.session_state.get('auto_desc', st.session_state.get('loaded_desc', "장인의 손길로 완성된 탄력 있는 붓모..."))
description = st.text_area("✍️ 메인 상세 설명", value=desc_default, height=150)

if st.session_state.get('auto_tags'):
    st.markdown("**🏷️ 추천 해시태그 (우측 상단 아이콘을 눌러 복사하세요)**")
    st.code(st.session_state.auto_tags, language="text")

# ===========================
# 2. 스토리 블록 (중간 사진 + 글)
# ===========================
st.header("2. 스토리 상세 블록 (최대 5개)")
st.markdown("사진과 글을 번갈아가며 배치하여 읽기 좋은 스토리텔링 상세페이지를 만드세요.")

loaded_story_blocks = st.session_state.get('loaded_story_blocks', [])
story_files = []
story_texts = []

if 'story_ai_blocks' not in st.session_state:
    st.session_state.story_ai_blocks = [None] * 5

def clear_story_ai(idx):
    if 'story_ai_blocks' in st.session_state and len(st.session_state.story_ai_blocks) > idx:
        st.session_state.story_ai_blocks[idx] = None

for i in range(5):
    with st.expander(f"📖 스토리 블록 {i+1} (선택사항)", expanded=(i==0)):
        c1, c2 = st.columns([1, 2])
        
        loaded_b64 = ""
        loaded_mime = ""
        loaded_txt = ""
        if i < len(loaded_story_blocks):
            loaded_b64 = loaded_story_blocks[i].get("b64", "")
            loaded_mime = loaded_story_blocks[i].get("mime", "")
            loaded_txt = loaded_story_blocks[i].get("text", "")
            
        with c1:
            f = st.file_uploader(f"블록 {i+1} 이미지", type=['jpg','png','jpeg'], key=f"img_widget_new_{i}", on_change=clear_story_ai, args=(i,))
            
            ai_info = st.session_state.story_ai_blocks[i]
            if ai_info and ai_info.get('b64'):
                import base64
                st.image(base64.b64decode(ai_info['b64']), width=450)
                st.success("✨ AI 사진 적용됨")
                if st.button("❌ 원본 복구", key=f"rev_{i}"):
                    clear_story_ai(i)
                    st.rerun()
                if st.button("🎨 세부 편집기 열기 (새창)", key=f"story_ai_edit_{i}", type="secondary"):
                    render_editor(f"story_{i}")
            elif f or loaded_b64:
                if f:
                    st.image(f, width=450)
                else:
                    import base64
                    st.image(base64.b64decode(loaded_b64), width=450)
                    st.info("✅ 보관함/AI 이미지 대기 중")
                    
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    keep_bg = st.checkbox("원본 배경 유지 (누끼 안 땀)", value=True, key=f"story_keep_{i}")
                with col_c2:
                    use_matting = st.checkbox("테두리 부드럽게 (털/복잡한 선)", value=False, disabled=keep_bg, key=f"story_mat_{i}")
                    
                if st.button("⚡ 바로 편집하기 (새창)", type="secondary", key=f"story_quick_{i}"):
                    with st.spinner("이미지 준비 중... ✂️"):
                        import io, PIL.Image
                        if f:
                            target_img = PIL.Image.open(f)
                        else:
                            import base64
                            target_img = PIL.Image.open(io.BytesIO(base64.b64decode(loaded_b64)))
                            
                        if keep_bg:
                            fg = target_img.convert("RGBA")
                        else:
                            from rembg import remove
                            img_byte_arr = io.BytesIO()
                            target_img.save(img_byte_arr, format='PNG')
                            fg_bytes = remove(img_byte_arr.getvalue(), session=get_rembg_session(), post_process_mask=True, alpha_matting=use_matting)
                            fg = PIL.Image.open(io.BytesIO(fg_bytes)).convert("RGBA")
                        
                        bg = PIL.Image.new("RGBA", fg.size, (255, 255, 255, 0))
                        
                        st.session_state[f'story_{i}_fg_img'] = fg
                        st.session_state[f'story_{i}_bg_img'] = bg
                        
                        if 'story_ai_blocks' in st.session_state and len(st.session_state.story_ai_blocks) > i and st.session_state.story_ai_blocks[i]:
                            st.session_state.story_ai_blocks[i]['b64'] = None
                    render_editor(f"story_{i}")
            
            ai_b64 = ai_info['b64'] if ai_info and ai_info.get('b64') else None
            ai_mime = ai_info['mime'] if ai_info and ai_info.get('mime') else None
            story_files.append({"file": f, "b64": loaded_b64, "mime": loaded_mime, "ai_b64": ai_b64, "ai_mime": ai_mime})
            
        with c2:
            t = st.text_area(f"블록 {i+1} 텍스트", value=loaded_txt, height=150, key=f"txt_{i}_{st.session_state.get('load_timestamp', 'new')}", placeholder="예: 최고급 양모를 사용하여 탄력이 매우 뛰어나며...")
            story_texts.append(t)
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button(f"🤖 글 작성", key=f"ai_btn_{i}"):
                    if not f and not loaded_b64:
                        st.warning("먼저 이미지를 업로드해주세요!")
                    elif not api_key:
                        st.error("API 키를 사이드바에 입력해주세요!")
                    else:
                        with st.spinner("AI가 분석 중입니다..."):
                            try:
                                genai.configure(api_key=api_key)
                                if f:
                                    img = PIL.Image.open(f)
                                else:
                                    import base64
                                    img = PIL.Image.open(io.BytesIO(base64.b64decode(loaded_b64)))
                                
                                web_context = st.session_state.get('last_web_info', '')
                                if web_context:
                                    prompt = f"상품 사진과 관련 정보야.\n[정보]\n{web_context}\n위 [정보]에 나오는 상품의 상징과 효능(예: 액막이, 재물운 등)을 반드시 활용해 줘! 길고 지루한 설명은 다 빼고, 전체 글자수 50자 이내로 짧고 임팩트 있는 카피라이팅을 2~3줄로 작성해 줘. (예: 나쁜 기운은 막고, 곁에는 행운만. 당신을 지켜주는 붉은 수호석.)"
                                else:
                                    prompt = "이 상품 사진의 핵심 디테일과 감성을 살려서, 길고 지루한 설명은 다 빼고 전체 글자수 50자 이내로 짧고 임팩트 있는 카피라이팅을 2~3줄로 작성해 줘. 고객의 시선을 확 사로잡을 수 있도록 간결하게 써 줘. (예: 시선을 끄는 붉은 빛. 완벽한 존재감.)"
                                try:
                                    model = genai.GenerativeModel('gemini-2.5-flash')
                                    response = model.generate_content([prompt, img])
                                except Exception:
                                    model = genai.GenerativeModel('gemini-flash-latest')
                                    response = model.generate_content([prompt, img])
                                
                                blocks = st.session_state.get('loaded_story_blocks', [])
                                while len(blocks) <= i: blocks.append({})
                                blocks[i]['text'] = response.text
                                blocks[i]['b64'] = loaded_b64
                                blocks[i]['mime'] = loaded_mime
                                if f:
                                    import base64
                                    bytes_data = f.getvalue()
                                    blocks[i]['b64'] = base64.b64encode(bytes_data).decode('utf-8')
                                    blocks[i]['mime'] = f.type
                                st.session_state.loaded_story_blocks = blocks
                                st.session_state.load_timestamp = str(datetime.datetime.now().timestamp())
                                st.rerun()
                            except Exception as e:
                                st.error(f"오류: {e}")
            with col_b2:
                if st.button(f"🎨 사진 고급화", key=f"ai_img_btn_{i}"):
                    if not f and not loaded_b64:
                        st.warning("먼저 이미지를 업로드해주세요!")
                    elif not api_key:
                        st.error("API 키를 사이드바에 입력해주세요!")
                    else:
                        with st.spinner("AI가 사진을 변환 중입니다...🎨"):
                            try:
                                genai.configure(api_key=api_key)
                                if f:
                                    img = PIL.Image.open(f)
                                else:
                                    import base64
                                    img = PIL.Image.open(io.BytesIO(base64.b64decode(loaded_b64)))
                                
                                try:
                                    # 1. 원본 피사체 배경 제거 (rembg)
                                    from rembg import remove
                                    img_byte_arr = io.BytesIO()
                                    img.save(img_byte_arr, format='PNG')
                                    fg_bytes = remove(img_byte_arr.getvalue(), session=get_rembg_session(), post_process_mask=True)
                                    fg_img = PIL.Image.open(io.BytesIO(fg_bytes)).convert("RGBA")
                                    fg_img = remove_small_noise(fg_img)
                                    
                                    # 2. 고급 배경 AI 생성 (4가지 테마)
                                    bg_prompts = [
                                        "A very elegant minimalist dark studio background for product photography, dramatic soft spotlight in the center, 8k resolution, completely empty, no text.",
                                        "A bright and airy minimalist studio background with soft natural morning light and gentle shadows, pure white marble surface, 8k resolution, completely empty, no text.",
                                        "A luxurious warm gold and beige studio background, soft bokeh, high-end product photography style, completely empty, 8k resolution, no text.",
                                        "A modern abstract geometric background in pastel tones, soft lighting, 3d render style, completely empty, perfect for product placement, no text."
                                    ]
                                    model = genai.GenerativeModel('models/gemini-2.5-flash-image')
                                    
                                    generated_bgs = []
                                    import concurrent.futures
                                    def generate_single_bg(prompt):
                                        pass # removed import os to fix UnboundLocalError
                                        engine = st.session_state.get('ai_engine', 'Photoroom API')
                                        img_data = None
                                        
                                        if engine == "Photoroom API" or engine.startswith("Photoroom"):
                                            pr_key = ""
                                            if os.path.exists("photoroom_key.txt"):
                                                with open("photoroom_key.txt", "r", encoding="utf-8") as f: pr_key = f.read().strip()
                                            if pr_key:
                                                img_data = generate_photoroom_bg(prompt, fg_img, pr_key)
                                                if img_data and isinstance(img_data, dict):
                                                    st.session_state['story_replicate_error'] = img_data.get("error", "Unknown error")
                                                    img_data = None
                                        
                                        if (img_data is None) and (engine == "Replicate API" or engine.startswith("Replicate") or engine.startswith("Photoroom")):
                                            rep_key = ""
                                            if os.path.exists("replicate_key.txt"):
                                                with open("replicate_key.txt", "r", encoding="utf-8") as f: rep_key = f.read().strip()
                                            if rep_key:
                                                img_data = generate_replicate_bg(prompt, fg_img, rep_key)
                                                if img_data and isinstance(img_data, dict):
                                                    st.session_state['story_replicate_error'] = img_data.get("error", "Unknown error")
                                                    img_data = None
                                                    
                                        if img_data is None:
                                            try:
                                                res = model.generate_content(prompt)
                                                img_data = res.candidates[0].content.parts[0].inline_data.data
                                            except Exception:
                                                img_data = None
                                        return img_data
                                            
                                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                                        results = executor.map(generate_single_bg, bg_prompts)
                                        
                                    for b_data in results:
                                        if b_data:
                                            generated_bgs.append(b_data)
                                            
                                    # 항상 4개의 배경을 보장 (API 필터링/에러 대비 파이썬 직접 렌더링)
                                    from PIL import ImageDraw
                                    def create_fallback_bg(theme_idx, size):
                                        bg = PIL.Image.new("RGBA", size, (255, 255, 255, 255))
                                        draw = ImageDraw.Draw(bg)
                                        if theme_idx == 0:
                                            for y in range(size[1]):
                                                ratio = y / size[1]
                                                r, g, b = int(58 - 45 * ratio), int(62 - 48 * ratio), int(71 - 54 * ratio)
                                                draw.line([(0, y), (size[0], y)], fill=(r, g, b, 255))
                                        elif theme_idx == 1:
                                            for y in range(size[1]):
                                                ratio = y / size[1]
                                                r, g, b = int(255 - 20 * ratio), int(255 - 20 * ratio), int(255 - 15 * ratio)
                                                draw.line([(0, y), (size[0], y)], fill=(r, g, b, 255))
                                        elif theme_idx == 2:
                                            for y in range(size[1]):
                                                ratio = y / size[1]
                                                r, g, b = int(212 - 50 * ratio), int(175 - 40 * ratio), int(55 - 10 * ratio)
                                                draw.line([(0, y), (size[0], y)], fill=(r, g, b, 255))
                                        else:
                                            for y in range(size[1]):
                                                ratio = y / size[1]
                                                r, g, b = int(255 - 15 * ratio), int(230 - 15 * ratio), int(230 - 20 * ratio)
                                                draw.line([(0, y), (size[0], y)], fill=(r, g, b, 255))
                                        return bg
                                        
                                    while len(generated_bgs) < 4:
                                        idx = len(generated_bgs)
                                        fallback = global_create_fallback_bg(idx, fg_img.size)
                                        out_bytes = io.BytesIO()
                                        fallback.save(out_bytes, format='PNG')
                                        generated_bgs.append(out_bytes.getvalue())
                                        
                                    # 첫번째 배경을 기본 배경으로 설정
                                    bg_img = PIL.Image.open(io.BytesIO(generated_bgs[0])).convert("RGBA")
                                    bg_img = bg_img.resize(fg_img.size, PIL.Image.Resampling.LANCZOS)
                                    
                                    st.session_state[f'story_{i}_ai_bg_candidates'] = generated_bgs
                                    st.session_state[f'story_{i}_fg_img'] = fg_img
                                    st.session_state[f'story_{i}_bg_img'] = bg_img
                                    
                                    if 'story_replicate_error' in st.session_state:
                                        err = st.session_state['story_replicate_error']
                                        st.error(f"⚠️ Replicate API 결제 잔액 부족 또는 오류로 인해 무료 제미나이(Gemini)로 대체 생성되었습니다.\n\n(상세: {err})\n\n완벽한 인페인팅 착용샷을 원하시면 Replicate 크레딧을 충전해주세요!")
                                        del st.session_state['story_replicate_error']
                                        

                                    if 'story_ai_blocks' in st.session_state and len(st.session_state.story_ai_blocks) > i and st.session_state.story_ai_blocks[i]:
                                        st.session_state.story_ai_blocks[i]['b64'] = None
                                        
                                except Exception as e:
                                    st.error(f"합성 오류: {e}")
                                    st.stop()
                                
                                render_editor(f"story_{i}")
                            except Exception as e:
                                st.error(f"오류: {e}")

# ===========================
# 3. 하단 배송/고객센터 선택
# ===========================
st.header("3. 하단 배송/고객센터 선택")
options = ["옵션 A: 해송 (010-4506-0728)", "옵션 B: 일송 (010-5419-7048)"]
cs_default = st.session_state.get('loaded_cs_opt', options[0])
idx = options.index(cs_default) if cs_default in options else 0
cs_option = st.radio("적용할 고객센터 템플릿을 선택하세요:", options, index=idx, horizontal=True)

# ===========================
# 4. 생성 로직
# ===========================
if st.button("✨ 럭셔리 상세페이지 생성하기", type="primary"):
    has_hero = hero_file is not None or st.session_state.get('loaded_hero_b64')
    if not has_hero:
        st.warning("대표 이미지를 반드시 업로드해주세요!")
    else:
        if st.session_state.get('hero_ai_b64'):
            encoded_hero = st.session_state.hero_ai_b64
            mime_hero = st.session_state.hero_ai_mime
        elif hero_file:
            encoded_hero, mime_hero = optimize_image(hero_file)
        else:
            encoded_hero = st.session_state.get('loaded_hero_b64')
            mime_hero = st.session_state.get('loaded_mime_hero', 'image/jpeg')
            
        img_tags_hero = f'<div class="image-wrapper hero-img-wrapper"><img src="data:{mime_hero};base64,{encoded_hero}"></div>'
        
        # 스토리 블록 처리
        final_story_blocks = []
        html_story_blocks = ""
        for idx_block, (file_info, text_val) in enumerate(zip(story_files, story_texts)):
            b64_val = ""
            mime_val = ""
            if file_info.get("ai_b64"):
                b64_val = file_info["ai_b64"]
                mime_val = file_info["ai_mime"]
            elif file_info["file"]:
                b64_val, mime_val = optimize_image(file_info["file"])
            elif file_info["b64"]:
                b64_val = file_info["b64"]
                mime_val = file_info["mime"]
                
            if b64_val or text_val.strip():
                final_story_blocks.append({
                    "b64": b64_val,
                    "mime": mime_val,
                    "text": text_val
                })
                html_story_blocks += f'<div class="story-section">'
                if b64_val:
                    html_story_blocks += f'<div class="image-wrapper story-img"><img src="data:{mime_val};base64,{b64_val}"></div>'
                if text_val.strip():
                    html_story_blocks += f'<div class="story-text">{text_val}</div>'
                html_story_blocks += f'</div>'

        if "일송" in cs_option:
            cs_name = "일송"; cs_phone = "010-5419-7048"
        else:
            cs_name = "해송"; cs_phone = "010-4506-0728"

        html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>{product_name} 상세페이지</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ margin: 0; padding: 0; background-color: #111; font-family: 'Noto Serif KR', serif; color: #eee; display: flex; justify-content: center; }}
        .detail-container {{ width: 100%; max-width: 860px; background-color: #1a1a1a; box-shadow: 0 0 80px rgba(212,175,55,0.08); overflow: hidden; position: relative; z-index: 1; border: 1px solid #333; }}
        section {{ padding: 80px 40px; text-align: center; }}
        .image-wrapper {{ position: relative; width: 100%; border-radius: 12px; overflow: hidden; box-shadow: 0 15px 40px rgba(0,0,0,0.5); margin-bottom: 20px; border: 1px solid #333; }}
        .image-wrapper img {{ width: 100%; height: auto; display: block; }}
        
        .hero {{ background: linear-gradient(180deg, #222 0%, #1a1a1a 100%); padding: 100px 40px 0; border-bottom: 1px solid #2a2a2a; }}
        .hero .title {{ font-size: 3.5rem; font-weight: 700; margin-bottom: 20px; background: linear-gradient(45deg, #d4af37, #f3e5ab); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 2px 20px rgba(212,175,55,0.2); letter-spacing: -1px; }}
        .hero-img-wrapper {{ margin: 0 -20px; width: calc(100% + 40px); border-radius: 16px 16px 0 0; border: none; box-shadow: 0 -20px 50px rgba(0,0,0,0.6); }}
        .desc-text {{ font-size: 1.35rem; line-height: 2.2; color: #ccc; padding: 30px; white-space: pre-wrap; word-break: keep-all; letter-spacing: 0.5px; font-weight: 300; }}
        
        /* 스토리 블록 스타일 */
        .features {{ background-color: #1a1a1a; padding: 40px 40px 80px; position: relative; }}
        .features::before {{ content: ''; position: absolute; top: 0; left: 10%; right: 10%; height: 1px; background: linear-gradient(90deg, transparent, rgba(212,175,55,0.5), transparent); }}
        .story-section {{ margin-bottom: 80px; }}
        .story-section:last-child {{ margin-bottom: 0; }}
        .story-img {{ border-radius: 8px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.4); border: 1px solid #333; }}
        .story-text {{ font-size: 1.25rem; line-height: 2.1; color: #bbb; padding: 0 20px; text-align: left; white-space: pre-wrap; word-break: keep-all; border-left: 3px solid rgba(212,175,55,0.8); font-weight: 300; }}
        
        .footer {{ background-color: #141414; padding: 70px 50px; text-align: left; border-top: 1px solid #2a2a2a; }}
        .policy-box {{ background: #1c1c1c; padding: 40px; border-radius: 16px; margin-bottom: 40px; box-shadow: 0 5px 30px rgba(0,0,0,0.4); border: 1px solid #2a2a2a; line-height: 1.9; color: #aaa; font-weight: 300; }}
        .policy-box h3 {{ color: #d4af37; font-size: 1.4rem; margin-top:0; letter-spacing: 1px; font-weight: 600; }}
        .policy-box b {{ color: #ddd; font-weight: 600; }}
        
        .cs-info {{ display: flex; justify-content: space-between; align-items: center; }}
        .logo-text {{ font-size: 2.8rem; font-weight: 700; color: #d4af37; letter-spacing: 2px; }}
        .cs-details {{ text-align: right; color: #888; line-height: 1.6; font-weight: 300; }}
        .cs-number {{ font-size: 1.6rem; font-weight: 700; color: #d4af37; margin-bottom: 8px; }}
    </style>
</head>
<body>
    <div class="detail-container">
        <section class="hero">
            <h1 class="title">{product_name}</h1>
            <div class="desc-text">{description}</div>
            {img_tags_hero}
        </section>

        {f'<section class="features">{html_story_blocks}</section>' if html_story_blocks else ''}

        <section class="footer">
            <div class="policy-box">
                <h3>배송 및 교환/환불 규정 안내</h3>
                <p>1. 평일 오후 2시 이전 주문 및 결제건, 당일 발송됩니다. (1-3일내 도착)<br>
                2. 배송은 <b>대한통운</b>으로 발송됩니다.<br>
                3. 고객님의 부재, 연락처 오류 등으로 인한 반송 시 왕복 택배비는 고객님 부담입니다.<br><br>
                <b>교환 및 반품 안내</b><br>
                1. 구매자의 단순 변심은 상품 수령일 후 7일 이내 가능합니다.<br>
                2. 제품 포장을 개봉하였거나 훼손된 경우 불가합니다.</p>
            </div>
            <div class="cs-info">
                <div class="brand"><span class="logo-text">{cs_name}</span></div>
                <div class="cs-details">
                    <p class="cs-number">고객센터 : {cs_phone}</p>
                    <p>상담시간 : 평일 10:00~18:00 / 주말 공휴일 휴무</p>
                    <p>주소 : 경기도 부천시 원미구 소사로 276번길 21</p>
                </div>
            </div>
        </section>
    </div>

    <!-- 이미지 다운로드 버튼 영역 (화면 고정) -->
    <div id="downloadBtnContainer" style="position: fixed; bottom: 30px; right: 30px; display: flex; gap: 15px; z-index: 9999;">
        <button id="downloadPngBtn" style="padding: 18px 25px; background: #2c3e2f; color: white; border: none; border-radius: 50px; font-size: 16px; font-weight: bold; cursor: pointer; box-shadow: 0 5px 20px rgba(0,0,0,0.4); transition: transform 0.2s ease;">
            📸 PNG 다운로드
        </button>
        <button id="downloadJpgBtn" style="padding: 18px 25px; background: #4a634e; color: white; border: none; border-radius: 50px; font-size: 16px; font-weight: bold; cursor: pointer; box-shadow: 0 5px 20px rgba(0,0,0,0.4); transition: transform 0.2s ease;">
            📸 JPG 다운로드
        </button>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    <script>
        function downloadImage(format) {{
            var btnId = format === 'png' ? 'downloadPngBtn' : 'downloadJpgBtn';
            var btn = document.getElementById(btnId);
            var otherBtn = format === 'png' ? document.getElementById('downloadJpgBtn') : document.getElementById('downloadPngBtn');
            
            var originalText = btn.innerHTML;
            btn.innerHTML = '⏳ 생성 중...';
            btn.style.opacity = '0.8';
            otherBtn.style.display = 'none';
            
            setTimeout(function() {{
                html2canvas(document.querySelector('.detail-container'), {{
                    useCORS: true,
                    scale: 1,
                    backgroundColor: '#111111'
                }}).then(canvas => {{
                    var link = document.createElement('a');
                    var mimeType = format === 'png' ? 'image/png' : 'image/jpeg';
                    var extension = format === 'png' ? '.png' : '.jpg';
                    var quality = format === 'png' ? undefined : 0.8;
                    
                    link.download = '{product_name}_상세페이지' + extension;
                    link.href = canvas.toDataURL(mimeType, quality);
                    link.click();
                    
                    btn.innerHTML = originalText;
                    btn.style.opacity = '1';
                    otherBtn.style.display = 'block';
                }}).catch(err => {{
                    alert('이미지 생성에 실패했습니다.');
                    btn.innerHTML = originalText;
                    btn.style.opacity = '1';
                    otherBtn.style.display = 'block';
                }});
            }}, 300);
        }}

        document.getElementById('downloadPngBtn').addEventListener('click', function() {{ downloadImage('png'); }});
        document.getElementById('downloadJpgBtn').addEventListener('click', function() {{ downloadImage('jpg'); }});
    </script>
</body>
</html>
        """
        
        md_story = ""
        for b in final_story_blocks:
            if b['text']:
                md_story += f"\n{b['text']}\n"
        
        md_content = f"""# {product_name}\n\n{description}\n\n{md_story}\n\n---\n## 🚚 배송 및 교환/환불 규정 안내\n- 평일 오후 2시 이전 주문 및 결제건, 당일 발송됩니다. (1-3일내 도착)\n- 배송은 **대한통운**으로 발송됩니다.\n- 교환 및 반품: 구매자의 단순 변심은 상품 수령일 후 7일 이내 가능합니다.\n\n### 📞 고객센터 ({cs_name})\n- **고객센터:** {cs_phone}\n- **상담시간:** 평일 10:00~18:00 / 주말 공휴일 휴무\n- **주소:** 경기도 부천시 원미구 소사로 276번길 21\n"""
        
        st.session_state['gen_data'] = {
            'product_name': product_name,
            'description': description,
            'search_keyword': search_keyword,
            'cs_option': cs_option,
            'encoded_hero': encoded_hero,
            'mime_hero': mime_hero,
            'story_blocks': final_story_blocks,
            'html_template': html_template,
            'md_content': md_content
        }
        st.balloons()

if 'gen_data' in st.session_state:
    data = st.session_state['gen_data']

    st.success("🎉 생성 완료! 아래쪽에서 [현재 작업물 내 PC에 보관하기] 버튼을 눌러 작업을 영구 보존할 수 있습니다.")
    
    if st.button("📁 현재 작업물 내 PC에 보관하기 (사이드바에 저장됨)", type="secondary"):
        saved_name = save_product(data['product_name'], data['description'], data['search_keyword'], data['cs_option'], data['encoded_hero'], data['mime_hero'], data['story_blocks'])
        st.success(f"'{saved_name}' 이름으로 안전하게 보관되었습니다! 왼쪽 메뉴에서 언제든 다시 열어볼 수 있습니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("🌐 HTML 원본 파일 다운로드", data['html_template'], file_name=f"{data['product_name']}_상세페이지.html", mime="text/html")
    with col2:
        st.download_button("📝 마크다운(MD) 텍스트 다운로드", data['md_content'], file_name=f"{data['product_name']}_상세페이지.md", mime="text/markdown")
    
    st.components.v1.html(data['html_template'], height=1000, scrolling=True)
