import streamlit as st
import base64
import os
import io
import json
import datetime
import PIL.Image
from duckduckgo_search import DDGS
import google.generativeai as genai

st.set_page_config(page_title="프리미엄 상세페이지 생성기", layout="wide")

@st.cache_resource
def get_rembg_session():
    from rembg import new_session
    return new_session('birefnet-general')

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

def save_product(name, desc, search_kw, cs_opt, hero_b64, mime_hero, story_blocks):
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
if st.sidebar.button("내 컴퓨터에 키 영구 저장하기"):
    if api_key:
        with open(api_key_file, "w") as f:
            f.write(api_key)
        st.sidebar.success("API 키 저장 완료!")

st.sidebar.markdown("---")
st.sidebar.header("📁 보관함 (저장된 제품)")
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
# 1. 메인 타이틀 & 대표 이미지
# ===========================
st.header("1. 메인 타이틀 & 대표 이미지")
st.markdown("**상세페이지 최상단에 배치될 대표 얼굴입니다.**")

def on_hero_change():
    st.session_state.pop('hero_ai_b64', None)
    st.session_state.pop('hero_ai_mime', None)

@st.dialog("🎨 세부 편집기 (팝업창)", width="large")
def render_hero_editor():
    if 'hero_fg_img' in st.session_state and 'hero_bg_img' in st.session_state:
        st.markdown("**[조작 방법]** 슬라이더나 텍스트를 변경하면 아래 결과 이미지가 실시간으로 업데이트됩니다.")
        
        # 레이아웃 2분할 (좌: 미리보기, 우: 컨트롤러)
        main_col1, main_col2 = st.columns([1, 1.2])
        
        with main_col1:
            # 미리보기 이미지를 그릴 공간
            preview_container = st.empty()
            
        with main_col2:
            tab1, tab2, tab3 = st.tabs(["🎯 피사체 조절", "🖼️ 배경 변경", "✍️ 글씨 오버레이"])
            
            with tab1:
                scale = st.slider("피사체 크기 조절 (배율)", 0.1, 2.0, 1.0, 0.05, key='edit_scale')
                offset_x = st.slider("가로 위치 (X)", -1000, 1000, 0, 10, key='edit_x')
                offset_y = st.slider("세로 위치 (Y)", -1000, 1000, 0, 10, key='edit_y')
                erode_size = st.slider("테두리 색번짐 제거 (픽셀 깎기)", 0, 10, 0, 1, key='edit_erode')
                
            with tab2:
                bg_type = st.radio("배경 설정 방식", ["기존 AI 배경 유지", "단색 배경 적용", "직접 이미지 업로드"], horizontal=True, key="edit_bg_type")
                
                bg_upload = None
                use_solid_bg = False
                ai_bg_index = 0
                
                if bg_type == "직접 이미지 업로드":
                    bg_upload = st.file_uploader("배경 사진 직접 업로드", type=['png','jpg','jpeg'], key='edit_bg_file')
                elif bg_type == "단색 배경 적용":
                    use_solid_bg = True
                    bg_color = st.color_picker("단색 배경 색상 선택", "#000000", key='edit_bg_color')
                else:
                    bg_color = "#000000"
                    if 'hero_ai_bg_candidates' in st.session_state and len(st.session_state.hero_ai_bg_candidates) > 0:
                        theme_names = ["1. 다크 스튜디오", "2. 밝은 대리석", "3. 골드 & 베이지", "4. 파스텔 기하학"]
                        opts = theme_names[:len(st.session_state.hero_ai_bg_candidates)]
                        selected_theme = st.selectbox("✨ 4가지 AI 배경 중 선택", opts, key='edit_ai_bg_idx')
                        ai_bg_index = opts.index(selected_theme)
                
            with tab3:
                overlay_text = st.text_input("삽입할 문구", "", key='edit_text')
                
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    font_family = st.selectbox("글꼴", ["나눔고딕", "나눔명조", "검은고딕 (매우 굵음)"], key='edit_font')
                with col_f2:
                    if font_family == "검은고딕 (매우 굵음)":
                        font_weight = st.selectbox("굵기", ["보통"], key='edit_weight')
                    else:
                        font_weight = st.selectbox("굵기", ["보통", "굵게"], key='edit_weight')
                        
                text_size = st.slider("글씨 크기", 10, 300, 80, 5, key='edit_text_size')
                text_x = st.slider("글씨 가로 위치 (좌우 이동)", -1000, 1000, 0, 10, key='edit_text_x')
                text_y = st.slider("글씨 세로 위치 (상하 이동)", 0, 1500, 100, 10, key='edit_text_y')
                text_color = st.color_picker("글씨 색상", "#FFFFFF", key='edit_color')
                
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
            
            fg = st.session_state.hero_fg_img.copy()
            bg = st.session_state.hero_bg_img.copy()
            
            if erode_size > 0:
                import cv2
                import numpy as np
                arr = np.array(fg)
                alpha = arr[:, :, 3]
                kernel = np.ones((erode_size, erode_size), np.uint8)
                alpha = cv2.erode(alpha, kernel, iterations=1)
                alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
                arr[:, :, 3] = alpha
                fg = PIL.Image.fromarray(arr)
            
            if bg_upload is not None:
                bg = PIL.Image.open(io.BytesIO(bg_upload.getvalue())).convert("RGBA")
                bg = bg.resize(st.session_state.hero_bg_img.size, PIL.Image.Resampling.LANCZOS)
            elif use_solid_bg:
                bg = PIL.Image.new("RGBA", bg.size, bg_color)
            else:
                if 'hero_ai_bg_candidates' in st.session_state and len(st.session_state.hero_ai_bg_candidates) > ai_bg_index:
                    bg = PIL.Image.open(io.BytesIO(st.session_state.hero_ai_bg_candidates[ai_bg_index])).convert("RGBA")
                    # 배경을 기존 캔버스 사이즈에 맞춤 (안전장치)
                    bg = bg.resize(st.session_state.hero_bg_img.size, PIL.Image.Resampling.LANCZOS)
            
            new_size = (int(fg.size[0] * scale), int(fg.size[1] * scale))
            if new_size[0] > 0 and new_size[1] > 0:
                fg = fg.resize(new_size, PIL.Image.Resampling.LANCZOS)
            
            shadow = PIL.Image.new("RGBA", new_size, (0, 0, 0, 0))
            shadow.paste((0, 0, 0, 180), (0, 0), mask=fg)
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=int(max(new_size)*0.01)))
            
            cx = (bg.size[0] - new_size[0]) // 2 + offset_x
            cy = (bg.size[1] - new_size[1]) // 2 + offset_y
            shadow_y = cy + int(max(new_size)*0.02)
            
            temp_layer = PIL.Image.new("RGBA", bg.size, (0, 0, 0, 0))
            temp_layer.paste(shadow, (cx, shadow_y), mask=shadow)
            temp_layer.paste(fg, (cx, cy), mask=fg)
            
            bg = PIL.Image.alpha_composite(bg, temp_layer)
            
            if overlay_text.strip():
                draw = ImageDraw.Draw(bg)
                
                font_map = {
                    "나눔고딕": {"보통": "fonts/NanumGothic-Regular.ttf", "굵게": "fonts/NanumGothic-Bold.ttf"},
                    "나눔명조": {"보통": "fonts/NanumMyeongjo-Regular.ttf", "굵게": "fonts/NanumMyeongjo-Bold.ttf"},
                    "검은고딕 (매우 굵음)": {"보통": "fonts/BlackHanSans-Regular.ttf", "굵게": "fonts/BlackHanSans-Regular.ttf"}
                }
                
                import os
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
                st.session_state.hero_ai_b64 = encoded
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
        if st.button("🎨 세부 편집기 열기 (새창)", type="secondary"):
            render_hero_editor()
        
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
                    from rembg import remove
                    img_bytes = hero_file.getvalue()
                    fg_bytes = remove(img_bytes, session=get_rembg_session(), post_process_mask=True, alpha_matting=use_matting)
                    fg = PIL.Image.open(io.BytesIO(fg_bytes)).convert("RGBA")
                
                # 기본 배경은 투명으로 설정 (나중에 단색/이미지로 변경 가능)
                bg = PIL.Image.new("RGBA", fg.size, (255, 255, 255, 0))
                
                st.session_state.hero_fg_img = fg
                st.session_state.hero_bg_img = bg
                st.session_state.hero_ai_b64 = None # 초기화
            render_hero_editor()
    elif st.session_state.get('loaded_hero_b64'):
        import base64
        st.image(base64.b64decode(st.session_state.loaded_hero_b64), width=450)
        st.info("✅ AI 생성/보관함 이미지가 대기 중입니다.")
with col2:
    search_kw_default = st.session_state.get('loaded_search_kw', "")
    search_keyword = st.text_input("🔍 인터넷 검색어 (예: 해송 세필붓)", value=search_kw_default)
    name_default = st.session_state.get('loaded_name', "소나무붓 해송(海松) 4종 세트")
    product_name = st.text_input("💎 상품명", value=name_default)

if hero_file is not None or st.session_state.get('loaded_hero_b64'):
    col_btn1, col_btn2 = st.columns(2)
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
                                    results = ddgs.text(f"{search_keyword}", max_results=5)
                                    for res in results:
                                        web_info += f"- {res['title']}: {res['body']}\n"
                            except Exception:
                                pass
                                
                        genai.configure(api_key=api_key)
                        if hero_file:
                            img = PIL.Image.open(hero_file)
                        else:
                            import base64
                            img = PIL.Image.open(io.BytesIO(base64.b64decode(st.session_state.loaded_hero_b64)))
                        
                        if web_info.strip():
                            prompt = f"상품 메인 대표 사진과 검색정보야.\n[정보]\n{web_info}\n요즘 트렌디한 인스타그램 쇼핑몰이나 애플(Apple) 광고처럼 아주 짧고 간결하면서도 임팩트 있는 카피라이팅을 2~3줄로 작성해 줘. 특징을 구체적으로 짚어주되, 구구절절 긴 문장은 절대 피하고, 감각적으로 줄바꿈을 해 줘."
                        else:
                            prompt = "이 사진의 특징과 디테일을 파악해서, 요즘 트렌디한 쇼핑몰이나 애플(Apple) 광고처럼 아주 짧고 간결하면서도 임팩트 있는 카피라이팅을 2~3줄로 작성해 줘. 구구절절 긴 문장은 절대 피하고, 시선을 확 끄는 핵심 단어 위주로 감각적인 줄바꿈을 해 줘."
                        
                        try:
                            model = genai.GenerativeModel('gemini-2.5-flash')
                            response = model.generate_content([prompt, img])
                        except Exception:
                            model = genai.GenerativeModel('gemini-flash-latest')
                            response = model.generate_content([prompt, img])
                        st.session_state.auto_desc = response.text
                        
                        if hero_file:
                            import base64
                            bytes_data = hero_file.getvalue()
                            st.session_state.loaded_hero_b64 = base64.b64encode(bytes_data).decode('utf-8')
                            st.session_state.loaded_hero_mime = hero_file.type
                        
                        st.session_state.load_timestamp = str(datetime.datetime.now().timestamp())
                        st.rerun()
                    except Exception as e:
                        st.error(f"오류: {e}")
    with col_btn2:
        if st.button("🎨 AI 스튜디오 사진 고급화"):
            if not api_key:
                st.error("👈 좌측 사이드바에 API 키를 입력해주세요!")
            else:
                with st.spinner("AI가 4가지 테마의 고급 배경을 생성 중입니다... (약 10~20초 소요) 🎨"):
                    try:
                        genai.configure(api_key=api_key)
                        if hero_file:
                            img = PIL.Image.open(hero_file)
                        else:
                            import base64
                            img = PIL.Image.open(io.BytesIO(base64.b64decode(st.session_state.loaded_hero_b64)))
                        
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
                                try:
                                    res = model.generate_content(prompt)
                                    return res.candidates[0].content.parts[0].inline_data.data
                                except Exception:
                                    return None
                                    
                            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                                results = executor.map(generate_single_bg, bg_prompts)
                                
                            for b_data in results:
                                if b_data:
                                    generated_bgs.append(b_data)
                            
                            # 항상 4개의 배경을 보장 (API 필터링/에러 대비 파이썬 직접 렌더링)
                            from PIL import ImageDraw
                            import io
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
                                        r, g, b = int(245 - 30 * ratio), int(230 - 30 * ratio), int(200 - 40 * ratio)
                                        draw.line([(0, y), (size[0], y)], fill=(r, g, b, 255))
                                else:
                                    for y in range(size[1]):
                                        ratio = y / size[1]
                                        r, g, b = int(230 - 10 * ratio), int(240 - 20 * ratio), int(255 - 10 * ratio)
                                        draw.line([(0, y), (size[0], y)], fill=(r, g, b, 255))
                                return bg
                            
                            while len(generated_bgs) < 4:
                                fb = create_fallback_bg(len(generated_bgs), fg_img.size)
                                out_b = io.BytesIO()
                                fb.save(out_b, format='PNG')
                                generated_bgs.append(out_b.getvalue())
                                
                            st.session_state.hero_ai_bg_candidates = generated_bgs
                            bg_img = PIL.Image.open(io.BytesIO(generated_bgs[0])).convert("RGBA")
                            
                            # 3. 배경을 원본 사진과 완전히 동일한 크기/비율로 맞추기
                            fg_ratio = fg_img.size[0] / fg_img.size[1]
                            bg_ratio = bg_img.size[0] / bg_img.size[1]
                            
                            if fg_ratio > bg_ratio:
                                new_bg_h = int(bg_img.size[0] / fg_ratio)
                                top = (bg_img.size[1] - new_bg_h) // 2
                                bg_img = bg_img.crop((0, top, bg_img.size[0], top + new_bg_h))
                            else:
                                new_bg_w = int(bg_img.size[1] * fg_ratio)
                                left = (bg_img.size[0] - new_bg_w) // 2
                                bg_img = bg_img.crop((left, 0, left + new_bg_w, bg_img.size[1]))
                                
                            bg_img = bg_img.resize(fg_img.size, PIL.Image.Resampling.LANCZOS)
                            
                            st.session_state.hero_fg_img = fg_img.copy()
                            st.session_state.hero_bg_img = bg_img.copy()
                            
                            # 그림자 생성 (원본 크기에 맞춤)
                            from PIL import ImageFilter
                            shadow = PIL.Image.new("RGBA", fg_img.size, (0, 0, 0, 0))
                            shadow.paste((0, 0, 0, 180), (0, 0), mask=fg_img)
                            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=int(max(fg_img.size)*0.01)))
                            
                            offset_y = int(max(fg_img.size)*0.02)
                            bg_img.paste(shadow, (0, offset_y), mask=shadow)
                            
                            # 원본 피사체를 원래 위치 그대로(0,0) 합성
                            bg_img.paste(fg_img, (0, 0), mask=fg_img)
                            
                            out_bytes = io.BytesIO()
                            bg_img.convert("RGB").save(out_bytes, format='PNG')
                            generated_bytes = out_bytes.getvalue()
                            mime_type = "image/png"
                        except Exception as e:
                            st.error(f"합성 오류: {e}")
                            st.stop()
                        import base64
                        st.session_state.hero_ai_b64 = base64.b64encode(generated_bytes).decode('utf-8')
                        st.session_state.hero_ai_mime = mime_type
                        st.rerun()
                    except Exception as e:
                        st.error(f"오류: {e}")

desc_default = st.session_state.get('auto_desc', st.session_state.get('loaded_desc', "장인의 손길로 완성된 탄력 있는 붓모..."))
description = st.text_area("✍️ 메인 상세 설명", value=desc_default, height=150)

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
            elif f:
                st.image(f, width=450)
            elif loaded_b64:
                import base64
                st.image(base64.b64decode(loaded_b64), width=450)
                st.info("✅ 보관함/AI 이미지 대기 중")
            
            ai_b64 = ai_info['b64'] if ai_info else None
            ai_mime = ai_info['mime'] if ai_info else None
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
                                
                                prompt = "이 상품 사진의 핵심 디테일과 감성을 살려서, 트렌디한 쇼핑몰 상세페이지에 들어갈 아주 짧고 임팩트 있는 카피라이팅 2~3줄을 작성해 줘. 길고 지루한 설명은 빼고, 고객의 시선을 확 사로잡을 수 있도록 간결하게 쓰고 감각적으로 줄바꿈을 넣어 줘."
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
                                    
                                    # 2. 고급 배경 AI 생성 (피사체 없이 배경만)
                                    bg_prompt = "A very elegant minimalist dark studio background for product photography, dramatic soft spotlight in the center, 8k resolution, completely empty, no text."
                                    model = genai.GenerativeModel('models/gemini-2.5-flash-image')
                                    response = model.generate_content(bg_prompt)
                                    
                                    generated_bytes = response.candidates[0].content.parts[0].inline_data.data
                                    if not generated_bytes:
                                        # 필터링 당하면 에러 대신 고급스러운 수제 그라데이션 배경 사용
                                        from PIL import ImageDraw, ImageFilter
                                        bg_img = PIL.Image.new("RGBA", fg_img.size, (15, 15, 18, 255))
                                        draw = ImageDraw.Draw(bg_img)
                                        for y in range(fg_img.size[1]):
                                            ratio = y / fg_img.size[1]
                                            r = int(58 - (58 - 13) * ratio)
                                            g = int(62 - (62 - 14) * ratio)
                                            b = int(71 - (71 - 17) * ratio)
                                            draw.line([(0, y), (fg_img.size[0], y)], fill=(r, g, b, 255))
                                    else:
                                        bg_img = PIL.Image.open(io.BytesIO(generated_bytes)).convert("RGBA")
                                    
                                    # 3. 배경을 원본 사진과 완전히 동일한 크기/비율로 맞추기
                                    fg_ratio = fg_img.size[0] / fg_img.size[1]
                                    bg_ratio = bg_img.size[0] / bg_img.size[1]
                                    
                                    if fg_ratio > bg_ratio:
                                        new_bg_h = int(bg_img.size[0] / fg_ratio)
                                        top = (bg_img.size[1] - new_bg_h) // 2
                                        bg_img = bg_img.crop((0, top, bg_img.size[0], top + new_bg_h))
                                    else:
                                        new_bg_w = int(bg_img.size[1] * fg_ratio)
                                        left = (bg_img.size[0] - new_bg_w) // 2
                                        bg_img = bg_img.crop((left, 0, left + new_bg_w, bg_img.size[1]))
                                        
                                    bg_img = bg_img.resize(fg_img.size, PIL.Image.Resampling.LANCZOS)
                                    
                                    # 그림자 생성 (원본 크기에 맞춤)
                                    from PIL import ImageFilter
                                    shadow = PIL.Image.new("RGBA", fg_img.size, (0, 0, 0, 0))
                                    shadow.paste((0, 0, 0, 180), (0, 0), mask=fg_img)
                                    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=int(max(fg_img.size)*0.01)))
                                    
                                    offset_y = int(max(fg_img.size)*0.02)
                                    bg_img.paste(shadow, (0, offset_y), mask=shadow)
                                    
                                    # 원본 피사체를 원래 위치 그대로(0,0) 합성
                                    bg_img.paste(fg_img, (0, 0), mask=fg_img)
                                    
                                    out_bytes = io.BytesIO()
                                    bg_img.convert("RGB").save(out_bytes, format='PNG')
                                    generated_bytes = out_bytes.getvalue()
                                    mime_type = "image/png"
                                except Exception as e:
                                    st.error(f"합성 오류: {e}")
                                    st.stop()
                                
                                import base64
                                st.session_state.story_ai_blocks[i] = {'b64': base64.b64encode(generated_bytes).decode('utf-8'), 'mime': mime_type}
                                st.rerun()
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
