import os
from PIL import Image, ImageDraw, ImageFont

def get_fonts():
    font_b = "fonts/BlackHanSans-Regular.ttf" if os.path.exists("fonts/BlackHanSans-Regular.ttf") else "malgun.ttf"
    font_r = "fonts/NanumGothic-Regular.ttf" if os.path.exists("fonts/NanumGothic-Regular.ttf") else "malgun.ttf"
    return font_b, font_r

def get_emoji_font(fallback):
    return "seguiemj.ttf" if os.path.exists("C:/Windows/Fonts/seguiemj.ttf") else fallback

def draw_text_with_fallback(draw, xy, text, font, fill, anchor=None):
    try:
        if anchor:
            draw.text(xy, text, font=font, fill=fill, anchor=anchor)
        else:
            draw.text(xy, text, font=font, fill=fill)
    except:
        draw.text(xy, text, font=font, fill=fill)

def render_template(bg, style_idx, tmpl_color, tmpl_title, tmpl_sub_top, tmpl_sub_bottom, points, shape_off_x=0, shape_off_y=0, text_off_x=0, text_off_y=0):
    W, H = bg.size
    
    shape_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    text_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    
    draw_s = ImageDraw.Draw(shape_layer)
    draw_t = ImageDraw.Draw(text_layer)
    
    font_b, font_r = get_fonts()
    font_e = get_emoji_font(font_r)
    
    if style_idx == 0:
        # 1. 클래식 하단 배너 (기존)
        banner_h = int(H * 0.16)
        banner_y = H - banner_h
        draw_s.rectangle([0, banner_y, W, H], fill=tmpl_color)
        
        f_icon = ImageFont.truetype(font_e, int(banner_h * 0.4))
        f_p_title = ImageFont.truetype(font_b, int(banner_h * 0.18))
        f_p_desc = ImageFont.truetype(font_r, int(banner_h * 0.12))
        
        col_w = W // 3
        for idx, (icon, title, desc) in enumerate(points):
            cx = idx * col_w + (col_w // 2)
            icon_x, text_x = cx - int(col_w * 0.25), cx - int(col_w * 0.05)
            draw_text_with_fallback(draw_t, (icon_x, banner_y + banner_h*0.5), icon, f_icon, "white", "mm")
            draw_text_with_fallback(draw_t, (text_x, banner_y + banner_h*0.35), title, f_p_title, "white", "lm")
            draw_text_with_fallback(draw_t, (text_x, banner_y + banner_h*0.65), desc, f_p_desc, "#DDDDDD", "lm")
            
        title_x, title_y = int(W * 0.08), int(H * 0.15)
        f_sub = ImageFont.truetype(font_b, int(W * 0.035))
        f_main = ImageFont.truetype(font_b, int(W * 0.09))
        
        draw_text_with_fallback(draw_t, (title_x, title_y), tmpl_sub_top, f_sub, "#222222")
        lines = tmpl_title.split("\n")
        curr_y = title_y + int(H * 0.05)
        for idx, line in enumerate(lines):
            try: bbox = draw_t.textbbox((title_x, curr_y), line, font=f_main)
            except: bbox = (title_x, curr_y, title_x + int(W*0.4), curr_y + int(H*0.08))
            if idx % 2 == 1:
                draw_s.rectangle([bbox[0]-10, bbox[1]-5, bbox[2]+10, bbox[3]+15], fill=tmpl_color)
                draw_t.text((title_x, curr_y), line, font=f_main, fill="white")
            else:
                draw_t.text((title_x, curr_y), line, font=f_main, fill="#222222")
            curr_y += int(H * 0.1)
        draw_t.text((title_x, curr_y + int(H * 0.02)), tmpl_sub_bottom, font=f_sub, fill="#333333")

    elif style_idx == 1:
        # 2. 모던 중앙 집중형 (인스타그램 스타일)
        overlay = Image.new("RGBA", bg.size, (0,0,0,120))
        bg = Image.alpha_composite(bg, overlay)
        
        f_main = ImageFont.truetype(font_b, int(W * 0.12))
        f_sub = ImageFont.truetype(font_b, int(W * 0.04))
        
        cy = int(H * 0.3)
        draw_text_with_fallback(draw_t, (W//2, cy), tmpl_sub_top, f_sub, tmpl_color, "mm")
        
        lines = tmpl_title.split("\n")
        cy += int(H * 0.08)
        for line in lines:
            draw_text_with_fallback(draw_t, (W//2, cy), line, f_main, "white", "mm")
            cy += int(H * 0.13)
            
        cy += int(H * 0.05)
        draw_text_with_fallback(draw_t, (W//2, cy), tmpl_sub_bottom, f_sub, "#DDDDDD", "mm")
        
        f_icon = ImageFont.truetype(font_e, int(H * 0.06))
        f_p_title = ImageFont.truetype(font_b, int(H * 0.03))
        col_w = W // 3
        banner_y = H - int(H * 0.15)
        for idx, (icon, title, desc) in enumerate(points):
            cx = idx * col_w + (col_w // 2)
            draw_text_with_fallback(draw_t, (cx, banner_y), icon, f_icon, "white", "mm")
            draw_text_with_fallback(draw_t, (cx, banner_y + int(H*0.05)), title, f_p_title, tmpl_color, "mm")

    elif style_idx == 2:
        # 3. 좌측 세로 리본
        ribbon_w = int(W * 0.35)
        draw_s.rectangle([0, 0, ribbon_w, H], fill=tmpl_color)
        
        f_sub = ImageFont.truetype(font_b, int(ribbon_w * 0.08))
        f_main = ImageFont.truetype(font_b, int(ribbon_w * 0.18))
        
        cx = ribbon_w // 2
        cy = int(H * 0.1)
        draw_text_with_fallback(draw_t, (cx, cy), tmpl_sub_top, f_sub, "#EEEEEE", "mm")
        
        lines = tmpl_title.split("\n")
        cy += int(H * 0.05)
        for line in lines:
            draw_text_with_fallback(draw_t, (cx, cy), line, f_main, "white", "mt")
            cy += int(H * 0.08)
            
        cy += int(H * 0.05)
        draw_text_with_fallback(draw_t, (cx, cy), tmpl_sub_bottom, f_sub, "#DDDDDD", "mm")
        
        cy += int(H * 0.15)
        f_icon = ImageFont.truetype(font_e, int(H * 0.05))
        f_p_title = ImageFont.truetype(font_b, int(H * 0.025))
        for (icon, title, desc) in points:
            draw_text_with_fallback(draw_t, (cx, cy), icon, f_icon, "white", "mm")
            draw_text_with_fallback(draw_t, (cx, cy + int(H*0.04)), title, f_p_title, "white", "mm")
            cy += int(H * 0.12)

    elif style_idx == 3:
        # 4. 우측 하단 미니멀 박스
        box_w, box_h = int(W * 0.45), int(H * 0.35)
        box_x, box_y = W - box_w - int(W*0.05), H - box_h - int(H*0.05)
        
        draw_s.rectangle([box_x, box_y, box_x+box_w, box_y+box_h], fill=(255,255,255,230))
        draw_s.rectangle([box_x, box_y, box_x+15, box_y+box_h], fill=tmpl_color)
        
        f_main = ImageFont.truetype(font_b, int(box_w * 0.15))
        f_sub = ImageFont.truetype(font_r, int(box_w * 0.06))
        
        tx, ty = box_x + 30, box_y + 20
        inline_title = tmpl_title.replace("\n", " ")
        draw_text_with_fallback(draw_t, (tx, ty), inline_title, f_main, "#222222", "lt")
        ty += int(box_h * 0.25)
        draw_text_with_fallback(draw_t, (tx, ty), tmpl_sub_top, f_sub, tmpl_color, "lt")
        
        ty += int(box_h * 0.15)
        f_p_title = ImageFont.truetype(font_b, int(box_w * 0.07))
        for (icon, title, desc) in points:
            draw_text_with_fallback(draw_t, (tx, ty), f"{icon} {title}", f_p_title, "#444444", "lt")
            ty += int(box_h * 0.15)

    elif style_idx == 4:
        # 5. 투컬럼 스플릿 (화면 분할)
        split_x = int(W * 0.4)
        draw_s.rectangle([0, 0, split_x, H], fill=tmpl_color)
        
        f_sub = ImageFont.truetype(font_b, int(split_x * 0.07))
        f_main = ImageFont.truetype(font_b, int(split_x * 0.18))
        f_p_title = ImageFont.truetype(font_b, int(split_x * 0.09))
        f_p_desc = ImageFont.truetype(font_r, int(split_x * 0.06))
        
        cx, cy = int(split_x * 0.1), int(H * 0.1)
        draw_text_with_fallback(draw_t, (cx, cy), tmpl_sub_top, f_sub, "#EEEEEE", "lt")
        
        lines = tmpl_title.split("\n")
        cy += int(H * 0.05)
        for line in lines:
            draw_text_with_fallback(draw_t, (cx, cy), line, f_main, "white", "lt")
            cy += int(H * 0.08)
            
        cy += int(H * 0.05)
        draw_text_with_fallback(draw_t, (cx, cy), tmpl_sub_bottom, f_sub, "#DDDDDD", "lt")
        
        cy += int(H * 0.15)
        for (icon, title, desc) in points:
            draw_text_with_fallback(draw_t, (cx, cy), f"{icon} {title}", f_p_title, "white", "lt")
            draw_text_with_fallback(draw_t, (cx, cy + int(H*0.03)), desc, f_p_desc, "#CCCCCC", "lt")
            cy += int(H * 0.1)

    elif style_idx == 5:
        # 6. 플로팅 포인트 카드
        card_h = int(H * 0.15)
        card_y = H - card_h - int(H*0.05)
        col_w = W // 3
        
        f_icon = ImageFont.truetype(font_e, int(card_h * 0.35))
        f_p_title = ImageFont.truetype(font_b, int(card_h * 0.2))
        f_p_desc = ImageFont.truetype(font_r, int(card_h * 0.12))
        
        for idx, (icon, title, desc) in enumerate(points):
            cx = idx * col_w + (col_w // 2)
            card_w = int(col_w * 0.85)
            left = cx - card_w//2
            draw_s.rounded_rectangle([left, card_y, left+card_w, card_y+card_h], radius=15, fill="white")
            
            draw_text_with_fallback(draw_t, (cx, card_y + card_h*0.2), icon, f_icon, "black", "mm")
            draw_text_with_fallback(draw_t, (cx, card_y + card_h*0.55), title, f_p_title, tmpl_color, "mm")
            draw_text_with_fallback(draw_t, (cx, card_y + card_h*0.8), desc, f_p_desc, "#555555", "mm")
            
        f_main = ImageFont.truetype(font_b, int(W * 0.1))
        f_sub = ImageFont.truetype(font_b, int(W * 0.04))
        
        cy = int(H * 0.1)
        draw_text_with_fallback(draw_t, (W//2, cy), tmpl_sub_top, f_sub, tmpl_color, "mm")
        inline_title = tmpl_title.replace("\n", " ")
        draw_text_with_fallback(draw_t, (W//2, cy + int(H*0.07)), inline_title, f_main, "#222222", "mm")

    elif style_idx == 6:
        # 7. 상하 분리형 (헤더&푸터)
        header_h = int(H * 0.18)
        footer_h = int(H * 0.12)
        
        draw_s.rectangle([0, 0, W, header_h], fill=tmpl_color)
        draw_s.rectangle([0, H - footer_h, W, H], fill="#222222")
        
        f_main = ImageFont.truetype(font_b, int(header_h * 0.4))
        f_sub = ImageFont.truetype(font_r, int(header_h * 0.2))
        
        inline_title = tmpl_title.replace("\n", " ")
        draw_text_with_fallback(draw_t, (W//2, header_h*0.4), inline_title, f_main, "white", "mm")
        draw_text_with_fallback(draw_t, (W//2, header_h*0.8), tmpl_sub_top, f_sub, "#DDDDDD", "mm")
        
        f_icon = ImageFont.truetype(font_e, int(footer_h * 0.4))
        f_p_title = ImageFont.truetype(font_b, int(footer_h * 0.25))
        
        col_w = W // 3
        footer_y = H - footer_h
        for idx, (icon, title, desc) in enumerate(points):
            cx = idx * col_w + (col_w // 2)
            draw_text_with_fallback(draw_t, (cx, footer_y + footer_h*0.3), icon, f_icon, "white", "mm")
            draw_text_with_fallback(draw_t, (cx, footer_y + footer_h*0.7), title, f_p_title, tmpl_color, "mm")

    elif style_idx == 7:
        # 8. 풀 오버레이 그라데이션
        grad_h = int(H * 0.4)
        grad_y = H - grad_h
        for i in range(grad_h):
            alpha = int((i / grad_h) * 220)
            draw_s.line([(0, grad_y + i), (W, grad_y + i)], fill=(0, 0, 0, alpha))
            
        f_main = ImageFont.truetype(font_b, int(W * 0.1))
        f_sub = ImageFont.truetype(font_r, int(W * 0.04))
        
        cy = grad_y + int(grad_h * 0.2)
        draw_text_with_fallback(draw_t, (W//2, cy), tmpl_sub_top, f_sub, tmpl_color, "mm")
        
        inline_title = tmpl_title.replace("\n", " ")
        cy += int(grad_h * 0.2)
        draw_text_with_fallback(draw_t, (W//2, cy), inline_title, f_main, "white", "mm")
        
        cy += int(grad_h * 0.3)
        col_w = W // 3
        f_p_title = ImageFont.truetype(font_b, int(grad_h * 0.08))
        for idx, (icon, title, desc) in enumerate(points):
            cx = idx * col_w + (col_w // 2)
            draw_text_with_fallback(draw_t, (cx, cy), f"{icon} {title}", f_p_title, "white", "mm")

    elif style_idx == 8:
        # 9. 대각선 스포트라이트
        draw_s.polygon([(0, 0), (int(W*0.6), 0), (int(W*0.4), H), (0, H)], fill=tmpl_color)
        
        title_x = int(W * 0.05)
        title_y = int(H * 0.1)
        
        f_sub = ImageFont.truetype(font_b, int(W * 0.035))
        f_main = ImageFont.truetype(font_b, int(W * 0.09))
        
        draw_text_with_fallback(draw_t, (title_x, title_y), tmpl_sub_top, f_sub, "#222222", "lt")
        lines = tmpl_title.split("\n")
        curr_y = title_y + int(H * 0.05)
        for line in lines:
            draw_text_with_fallback(draw_t, (title_x, curr_y), line, f_main, "white", "lt")
            curr_y += int(H * 0.1)
            
        curr_y += int(H * 0.1)
        f_p_title = ImageFont.truetype(font_b, int(W * 0.04))
        f_p_desc = ImageFont.truetype(font_r, int(W * 0.03))
        for (icon, title, desc) in points:
            draw_text_with_fallback(draw_t, (title_x, curr_y), f"{icon} {title}", f_p_title, "#222222", "lt")
            draw_text_with_fallback(draw_t, (title_x, curr_y + int(H*0.04)), desc, f_p_desc, "white", "lt")
            curr_y += int(H * 0.1)

    elif style_idx == 9:
        # 10. 원형 배지(Badge) 강조형
        badge_r = int(W * 0.2)
        bx, by = W - (badge_r*2) - int(W*0.05), int(H*0.05)
        draw_s.ellipse([bx, by, bx+badge_r*2, by+badge_r*2], fill=tmpl_color)
        draw_s.ellipse([bx+10, by+10, bx+badge_r*2-10, by+badge_r*2-10], outline="white", width=3)
        
        f_main = ImageFont.truetype(font_b, int(badge_r * 0.35))
        f_sub = ImageFont.truetype(font_b, int(badge_r * 0.15))
        
        cx, cy = bx + badge_r, by + badge_r
        inline_title = tmpl_title.replace("\n", " ")
        short_title = inline_title[:5] + (".." if len(inline_title)>5 else "")
        draw_text_with_fallback(draw_t, (cx, cy - int(badge_r*0.2)), short_title, f_main, "white", "mm")
        draw_text_with_fallback(draw_t, (cx, cy + int(badge_r*0.3)), tmpl_sub_top[:8], f_sub, "yellow", "mm")
        
        banner_h = int(H * 0.12)
        banner_y = H - banner_h
        draw_s.rectangle([0, banner_y, W, H], fill=(255,255,255,230))
        
        f_p_title = ImageFont.truetype(font_b, int(banner_h * 0.3))
        col_w = W // 3
        for idx, (icon, title, desc) in enumerate(points):
            ccx = idx * col_w + (col_w // 2)
            draw_text_with_fallback(draw_t, (ccx, banner_y + banner_h*0.5), f"{icon} {title}", f_p_title, tmpl_color, "mm")

    bg.paste(shape_layer, (shape_off_x, shape_off_y), mask=shape_layer)
    bg.paste(text_layer, (text_off_x, text_off_y), mask=text_layer)
    return bg
