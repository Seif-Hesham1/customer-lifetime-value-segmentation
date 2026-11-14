"""Tiny python-pptx helper: build clean 16:9 decks from a list of slide dicts."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image

BLUE, DARK, GREY = RGBColor(0x25, 0x63, 0xEB), RGBColor(0x0F, 0x17, 0x2A), RGBColor(0x64, 0x74, 0x8B)
W, H = Inches(13.333), Inches(7.5)


def _tb(slide, x, y, w, h, paras, size=20, color=DARK, bold=False, bullet=False, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h); tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for i, t in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = ('•  ' + t) if bullet else t
        r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = 'Calibri'
        p.space_after = Pt(10)
    return box


def build(path, deck_title, slides):
    prs = Presentation(); prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]
    for n, s in enumerate(slides, 1):
        sl = prs.slides.add_slide(blank)
        if s.get('kind') == 'title':
            bg = sl.shapes.add_shape(1, 0, 0, W, H); bg.fill.solid(); bg.fill.fore_color.rgb = DARK; bg.line.fill.background()
            _tb(sl, Inches(.8), Inches(2.2), Inches(11.5), Inches(1.8), [s['title']], 44, RGBColor(255, 255, 255), True)
            _tb(sl, Inches(.8), Inches(4.0), Inches(11.5), Inches(1.5), s['bullets'], 22, RGBColor(0x93, 0xC5, 0xFD))
            _tb(sl, Inches(.8), Inches(6.6), Inches(11), Inches(.5), ['NextGen Institute · Machine Learning · Final Project'], 14, RGBColor(0x94, 0xA3, 0xB8))
            continue
        bar = sl.shapes.add_shape(1, 0, 0, W, Inches(1.05)); bar.fill.solid(); bar.fill.fore_color.rgb = BLUE; bar.line.fill.background()
        _tb(sl, Inches(.5), Inches(.12), Inches(12.3), Inches(.85), [s['title']], 30, RGBColor(255, 255, 255), True, anchor=MSO_ANCHOR.MIDDLE)
        img, bullets = s.get('image'), s.get('bullets', [])
        top = Inches(1.35)
        if img:
            iw, ih = Image.open(img).size; ratio = iw / ih
            if s.get('layout') == 'stack':      # picture on top (wide), bullets below
                maxw, maxh = Inches(12.3), Inches(3.9)
                w = maxw if maxw / ratio <= maxh else int(maxh * ratio); h = int(w / ratio)
                sl.shapes.add_picture(img, int((W - w) / 2), top, w, h)
                _tb(sl, Inches(.6), top + h + Inches(.15), Inches(12.1), Inches(7.2) - top - h - Inches(.2), bullets, s.get('size', 18), bullet=True)
            else:                               # picture right, bullets left
                maxw, maxh = Inches(7.4), Inches(5.6)
                w = maxw if maxw / ratio <= maxh else int(maxh * ratio); h = int(w / ratio)
                sl.shapes.add_picture(img, W - w - Inches(.4), top, w, h)
                _tb(sl, Inches(.5), top, W - w - Inches(1.1), Inches(5.6), bullets, s.get('size', 20), bullet=True)
        else:
            _tb(sl, Inches(.7), top, Inches(11.9), Inches(5.6), bullets, s.get('size', 24), bullet=True)
        _tb(sl, Inches(.5), Inches(7.05), Inches(10), Inches(.35), [deck_title], 11, GREY)
        _tb(sl, Inches(12.2), Inches(7.05), Inches(.8), Inches(.35), [str(n)], 11, GREY)
        if s.get('notes'):
            sl.notes_slide.notes_text_frame.text = s['notes']
    prs.save(path)
