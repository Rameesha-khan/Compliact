"""
Face detection script using OpenCV.

Usage:
    python face_detection.py <image_path> [--output <output_path>]

Arguments:
    image_path          Path to the input image file.
    --output, -o        Path for the output image (default: <input>_detected.<ext>)
    --scale-factor      Detection scale factor (default: 1.1)
    --min-neighbors     Minimum neighbours per detection (default: 5)
    --min-size          Minimum face size as WxH (default: 30x30)
"""

import argparse
import io
import sys
from pathlib import Path

# Ensure stdin/stdout handle Unicode correctly on all platforms (esp. Windows CP1252).
# Cast to TextIOWrapper so type checkers recognise .reconfigure(); guarded by hasattr
# so it is safe on any implementation that doesn't expose the method.
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if isinstance(sys.stdin, io.TextIOWrapper):
    sys.stdin.reconfigure(encoding="utf-8")

import datetime

import cv2
import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image, ImageDraw, ImageFont

# Segoe UI is present on all modern Windows systems and supports full Unicode.
_PDF_FONT_REGULAR = "C:/Windows/Fonts/segoeui.ttf"
_PDF_FONT_BOLD    = "C:/Windows/Fonts/segoeuib.ttf"


# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------

# Every user-facing string lives here, keyed by language code then string key.
# Colour names map 1-to-1 with the English keys in _COLOUR_PALETTE below.
# yes_answers / no_answers list what the user may type for that language
# (Latin shortcuts y/n are always accepted too, see ask_yes_no).
TRANSLATIONS: dict[str, dict] = {
    "en": {
        "lang_name": "English",
        "select_prompt": "Choose a language / Choisissez une langue / Elige un idioma:\n"
                         "  en  English\n  fr  Français\n  es  Español\n"
                         "  ar  العربية\n  pt  Português\n  de  Deutsch\n"
                         "  ur  اردو\n"
                         "Your choice: ",
        "invalid_lang":  "Please type one of: en, fr, es, ar, pt, de, ur.",
        "detected_n":    "Detected {n} face(s).",
        "none_detected": "No faces detected.",
        "face_label":    "Face {i} of {total}",
        "face_label_1":  "A person",
        "consent_q":     "{face_label} detected — clothing colour looks {colour}. "
                         "Do you have their consent to share this photo? (yes/no): ",
        "consent_yes":   "  Got it — face {i} marked.",
        "blur_q":        "  No problem. Would you like to blur that face instead? (yes/no): ",
        "blur_no":       "  Okay — face {i} left as-is.",
        "blur_done":     "  Face {i} blurred ({style}).",
        "style_prompt":  "  Blur style [square (s), oval (o), strong (t), silhouette (l), emoji (e)]? ",
        "style_invalid": "  Please type square, oval, strong, silhouette, or emoji (or the letter).",
        "invalid_yn":    "Please answer yes or no.",
        "saved":         "\nOutput saved to: {path}",
        "summary":       "\n{total} face(s) processed: {consented} consented, "
                         "{blurred} blurred ({blur_breakdown}), {skipped} left as-is. "
                         "Ready to post!",
        "ai_watermark":  "AI-GENERATED CONTENT",
        "ai_label":      "⚠ AI-generated content detected (EU AI Act label applied).",
        "scene_indoor":  "Scene: indoors.",
        "scene_outdoor": "Scene: outdoors.",
        "score_line":    "Safe Content Score: {score}/100 ({risk}).",
        "score_safe":    "safe",
        "score_medium":  "medium risk",
        "score_high":    "high risk",
        "yes_answers":   ("yes", "y"),
        "no_answers":    ("no", "n"),
        "colours": {
            "red": "red", "orange": "orange", "yellow": "yellow",
            "green": "green", "teal": "teal", "blue": "blue",
            "purple": "purple", "pink": "pink", "white": "white",
            "light grey": "light grey", "grey": "grey", "dark grey": "dark grey",
            "black": "black", "brown": "brown", "beige": "beige",
        },
    },
    "fr": {
        "lang_name": "Français",
        "select_prompt": "",  # only shown in English
        "invalid_lang":  "",
        "detected_n":    "{n} visage(s) détecté(s).",
        "none_detected": "Aucun visage détecté.",
        "face_label":    "Visage {i} sur {total}",
        "face_label_1":  "Une personne",
        "consent_q":     "{face_label} détecté — couleur vestimentaire : {colour}. "
                         "Avez-vous le consentement de cette personne pour partager cette photo ? (oui/non) : ",
        "consent_yes":   "  Parfait — visage {i} marqué.",
        "blur_q":        "  Pas de souci. Voulez-vous flouter ce visage à la place ? (oui/non) : ",
        "blur_no":       "  D'accord — visage {i} laissé tel quel.",
        "blur_done":     "  Visage {i} flouté ({style}).",
        "style_prompt":  "  Style de floutage [carré (s), ovale (o), fort (t), silhouette (l), emoji (e)] ? ",
        "style_invalid": "  Veuillez choisir : carré, ovale, fort, silhouette ou emoji (ou la lettre).",
        "invalid_yn":    "Veuillez répondre par oui ou non.",
        "saved":         "\nImage enregistrée sous : {path}",
        "summary":       "\n{total} visage(s) traité(s) : {consented} avec consentement, "
                         "{blurred} flouté(s) ({blur_breakdown}), {skipped} laissé(s) tel quel. "
                         "Prêt(e) à publier !",
        "ai_watermark":  "CONTENU GÉNÉRÉ PAR IA",
        "ai_label":      "⚠ Contenu généré par IA détecté (étiquette Loi IA UE appliquée).",
        "scene_indoor":  "Scène : intérieur.",
        "scene_outdoor": "Scène : extérieur.",
        "score_line":    "Score de contenu sûr : {score}/100 ({risk}).",
        "score_safe":    "sûr",
        "score_medium":  "risque moyen",
        "score_high":    "risque élevé",
        "yes_answers":   ("oui", "o", "yes", "y"),
        "no_answers":    ("non", "n", "no"),
        "colours": {
            "red": "rouge", "orange": "orange", "yellow": "jaune",
            "green": "vert", "teal": "vert sarcelle", "blue": "bleu",
            "purple": "violet", "pink": "rose", "white": "blanc",
            "light grey": "gris clair", "grey": "gris", "dark grey": "gris foncé",
            "black": "noir", "brown": "marron", "beige": "beige",
        },
    },
    "es": {
        "lang_name": "Español",
        "select_prompt": "",
        "invalid_lang":  "",
        "detected_n":    "Se detectaron {n} cara(s).",
        "none_detected": "No se detectaron caras.",
        "face_label":    "Cara {i} de {total}",
        "face_label_1":  "Una persona",
        "consent_q":     "{face_label} detectada — color de ropa: {colour}. "
                         "¿Tienes el consentimiento de esta persona para compartir la foto? (sí/no): ",
        "consent_yes":   "  Entendido — cara {i} marcada.",
        "blur_q":        "  Sin problema. ¿Quieres difuminar esa cara en su lugar? (sí/no): ",
        "blur_no":       "  De acuerdo — cara {i} sin cambios.",
        "blur_done":     "  Cara {i} difuminada ({style}).",
        "style_prompt":  "  Estilo de difuminado [cuadrado (s), oval (o), intenso (t), silueta (l), emoji (e)]? ",
        "style_invalid": "  Por favor elige: cuadrado, oval, intenso, silueta o emoji (o la letra).",
        "invalid_yn":    "Por favor responde sí o no.",
        "saved":         "\nImagen guardada en: {path}",
        "summary":       "\n{total} cara(s) procesada(s): {consented} con consentimiento, "
                         "{blurred} difuminada(s) ({blur_breakdown}), {skipped} sin cambios. "
                         "¡Lista para publicar!",
        "ai_watermark":  "CONTENIDO GENERADO POR IA",
        "ai_label":      "⚠ Contenido generado por IA detectado (etiqueta Ley IA UE aplicada).",
        "scene_indoor":  "Escena: interior.",
        "scene_outdoor": "Escena: exterior.",
        "score_line":    "Puntuación de contenido seguro: {score}/100 ({risk}).",
        "score_safe":    "seguro",
        "score_medium":  "riesgo medio",
        "score_high":    "riesgo alto",
        "yes_answers":   ("sí", "si", "s", "yes", "y"),
        "no_answers":    ("no", "n"),
        "colours": {
            "red": "rojo", "orange": "naranja", "yellow": "amarillo",
            "green": "verde", "teal": "verde azulado", "blue": "azul",
            "purple": "morado", "pink": "rosa", "white": "blanco",
            "light grey": "gris claro", "grey": "gris", "dark grey": "gris oscuro",
            "black": "negro", "brown": "marrón", "beige": "beige",
        },
    },
    "ar": {
        "lang_name": "العربية",
        "select_prompt": "",
        "invalid_lang":  "",
        "detected_n":    "تم اكتشاف {n} وجه/وجوه.",
        "none_detected": "لم يتم اكتشاف أي وجوه.",
        "face_label":    "الوجه {i} من {total}",
        "face_label_1":  "شخص",
        "consent_q":     "{face_label} — لون الملابس يبدو {colour}. "
                         "هل لديك موافقة هذا الشخص لمشاركة الصورة؟ (نعم/لا): ",
        "consent_yes":   "  تمام — تم تحديد الوجه {i}.",
        "blur_q":        "  لا مشكلة. هل تريد تمويه هذا الوجه بدلاً من ذلك؟ (نعم/لا): ",
        "blur_no":       "  حسناً — الوجه {i} سيبقى كما هو.",
        "blur_done":     "  تم تمويه الوجه {i} ({style}).",
        "style_prompt":  "  أسلوب التمويه [مربع (s)، بيضاوي (o)، قوي (t)، صورة ظلية (l)، إيموجي (e)]؟ ",
        "style_invalid": "  الرجاء الاختيار: مربع، بيضاوي، قوي، صورة ظلية، إيموجي (أو الحرف).",
        "invalid_yn":    "الرجاء الإجابة بنعم أو لا.",
        "saved":         "\nتم حفظ الصورة في: {path}",
        "summary":       "\nتمت معالجة {total} وجه/وجوه: {consented} بموافقة، "
                         "{blurred} مموّه/مموّهة ({blur_breakdown})، {skipped} بدون تغيير. "
                         "جاهز للنشر!",
        "ai_watermark":  "محتوى مُنشأ بالذكاء الاصطناعي",
        "ai_label":      "⚠ تم اكتشاف محتوى مُنشأ بالذكاء الاصطناعي (تم تطبيق تسمية قانون الذكاء الاصطناعي الأوروبي).",
        "scene_indoor":  "المشهد: داخلي.",
        "scene_outdoor": "المشهد: خارجي.",
        "score_line":    "نقاط المحتوى الآمن: {score}/100 ({risk}).",
        "score_safe":    "آمن",
        "score_medium":  "خطر متوسط",
        "score_high":    "خطر مرتفع",
        "yes_answers":   ("نعم", "yes", "y"),
        "no_answers":    ("لا", "no", "n"),
        "colours": {
            "red": "أحمر", "orange": "برتقالي", "yellow": "أصفر",
            "green": "أخضر", "teal": "أزرق مخضر", "blue": "أزرق",
            "purple": "بنفسجي", "pink": "وردي", "white": "أبيض",
            "light grey": "رمادي فاتح", "grey": "رمادي", "dark grey": "رمادي غامق",
            "black": "أسود", "brown": "بني", "beige": "بيج",
        },
    },
    "pt": {
        "lang_name": "Português",
        "select_prompt": "",
        "invalid_lang":  "",
        "detected_n":    "{n} rosto(s) detectado(s).",
        "none_detected": "Nenhum rosto detectado.",
        "face_label":    "Rosto {i} de {total}",
        "face_label_1":  "Uma pessoa",
        "consent_q":     "{face_label} detectado — cor da roupa: {colour}. "
                         "Você tem o consentimento dessa pessoa para compartilhar a foto? (sim/não): ",
        "consent_yes":   "  Certo — rosto {i} marcado.",
        "blur_q":        "  Sem problema. Quer desfocar esse rosto? (sim/não): ",
        "blur_no":       "  Ok — rosto {i} mantido como está.",
        "blur_done":     "  Rosto {i} desfocado ({style}).",
        "style_prompt":  "  Estilo de desfoque [quadrado (s), oval (o), forte (t), silhueta (l), emoji (e)]? ",
        "style_invalid": "  Por favor escolha: quadrado, oval, forte, silhueta ou emoji (ou a letra).",
        "invalid_yn":    "Por favor responda sim ou não.",
        "saved":         "\nImagem salva em: {path}",
        "summary":       "\n{total} rosto(s) processado(s): {consented} com consentimento, "
                         "{blurred} desfocado(s) ({blur_breakdown}), {skipped} sem alteração. "
                         "Pronto para publicar!",
        "ai_watermark":  "CONTEÚDO GERADO POR IA",
        "ai_label":      "⚠ Conteúdo gerado por IA detectado (rótulo Lei IA UE aplicado).",
        "scene_indoor":  "Cena: interior.",
        "scene_outdoor": "Cena: exterior.",
        "score_line":    "Pontuação de conteúdo seguro: {score}/100 ({risk}).",
        "score_safe":    "seguro",
        "score_medium":  "risco médio",
        "score_high":    "risco alto",
        "yes_answers":   ("sim", "s", "yes", "y"),
        "no_answers":    ("não", "nao", "n", "no"),
        "colours": {
            "red": "vermelho", "orange": "laranja", "yellow": "amarelo",
            "green": "verde", "teal": "verde-azulado", "blue": "azul",
            "purple": "roxo", "pink": "rosa", "white": "branco",
            "light grey": "cinza claro", "grey": "cinza", "dark grey": "cinza escuro",
            "black": "preto", "brown": "marrom", "beige": "bege",
        },
    },
    "de": {
        "lang_name": "Deutsch",
        "select_prompt": "",
        "invalid_lang":  "",
        "detected_n":    "{n} Gesicht(er) erkannt.",
        "none_detected": "Keine Gesichter erkannt.",
        "face_label":    "Gesicht {i} von {total}",
        "face_label_1":  "Eine Person",
        "consent_q":     "{face_label} erkannt — Kleidungsfarbe: {colour}. "
                         "Liegt das Einverständnis dieser Person vor, das Foto zu teilen? (ja/nein): ",
        "consent_yes":   "  Alles klar — Gesicht {i} markiert.",
        "blur_q":        "  Kein Problem. Soll das Gesicht stattdessen unkenntlich gemacht werden? (ja/nein): ",
        "blur_no":       "  Okay — Gesicht {i} bleibt unverändert.",
        "blur_done":     "  Gesicht {i} unkenntlich gemacht ({style}).",
        "style_prompt":  "  Stil [quadrat (s), oval (o), stark (t), silhouette (l), emoji (e)]? ",
        "style_invalid": "  Bitte wählen: quadrat, oval, stark, silhouette oder emoji (oder den Buchstaben).",
        "invalid_yn":    "Bitte mit ja oder nein antworten.",
        "saved":         "\nBild gespeichert unter: {path}",
        "summary":       "\n{total} Gesicht(er) verarbeitet: {consented} mit Einverständnis, "
                         "{blurred} unkenntlich gemacht ({blur_breakdown}), {skipped} unverändert. "
                         "Bereit zum Teilen!",
        "ai_watermark":  "KI-GENERIERTER INHALT",
        "ai_label":      "⚠ KI-generierter Inhalt erkannt (EU-KI-Gesetz-Label angebracht).",
        "scene_indoor":  "Szene: drinnen.",
        "scene_outdoor": "Szene: draußen.",
        "score_line":    "Inhaltssicherheitsbewertung: {score}/100 ({risk}).",
        "score_safe":    "sicher",
        "score_medium":  "mittleres Risiko",
        "score_high":    "hohes Risiko",
        "yes_answers":   ("ja", "j", "yes", "y"),
        "no_answers":    ("nein", "n", "no"),
        "colours": {
            "red": "rot", "orange": "orange", "yellow": "gelb",
            "green": "grün", "teal": "blaugrün", "blue": "blau",
            "purple": "lila", "pink": "rosa", "white": "weiß",
            "light grey": "hellgrau", "grey": "grau", "dark grey": "dunkelgrau",
            "black": "schwarz", "brown": "braun", "beige": "beige",
        },
    },
    "ur": {
        "lang_name": "اردو",
        "select_prompt": "",
        "invalid_lang":  "",
        "detected_n":    "{n} چہرہ/چہرے پائے گئے۔",
        "none_detected": "کوئی چہرہ نہیں ملا۔",
        "face_label":    "چہرہ {i} از {total}",
        "face_label_1":  "ایک شخص",
        "consent_q":     "{face_label} ملا — لباس کا رنگ {colour} لگتا ہے۔ "
                         "کیا آپ کے پاس اس شخص کی اجازت ہے کہ یہ تصویر شیئر کریں؟ (ہاں/نہیں): ",
        "consent_yes":   "  بالکل — چہرہ {i} نشان زد کر دیا گیا۔",
        "blur_q":        "  کوئی بات نہیں۔ کیا آپ اس کی بجائے یہ چہرہ دھندلا کرنا چاہیں گے؟ (ہاں/نہیں): ",
        "blur_no":       "  ٹھیک ہے — چہرہ {i} جوں کا توں چھوڑ دیا گیا۔",
        "blur_done":     "  چہرہ {i} دھندلا کر دیا گیا ({style})۔",
        "style_prompt":  "  دھندلاپن کا انداز [مربع (s)، بیضوی (o)، مضبوط (t)، خاکہ (l)، ایموجی (e)]؟ ",
        "style_invalid": "  براہ کرم مربع، بیضوی، مضبوط، خاکہ یا ایموجی میں سے کوئی ایک چنیں (یا حرف)۔",
        "invalid_yn":    "براہ کرم ہاں یا نہیں میں جواب دیں۔",
        "saved":         "\nتصویر یہاں محفوظ کی گئی: {path}",
        "summary":       "\n{total} چہرہ/چہرے پروسیس ہوئے: {consented} کی اجازت ملی، "
                         "{blurred} دھندلے کیے گئے ({blur_breakdown})، {skipped} بغیر تبدیلی کے۔ "
                         "پوسٹ کے لیے تیار!",
        "ai_watermark":  "AI سے تیار کردہ مواد",
        "ai_label":      "⚠ AI سے تیار کردہ مواد ملا (EU AI Act لیبل لگایا گیا)۔",
        "scene_indoor":  "منظر: اندرونی۔",
        "scene_outdoor": "منظر: بیرونی۔",
        "score_line":    "محفوظ مواد اسکور: {score}/100 ({risk})۔",
        "score_safe":    "محفوظ",
        "score_medium":  "درمیانہ خطرہ",
        "score_high":    "زیادہ خطرہ",
        "yes_answers":   ("ہاں", "h", "yes", "y"),
        "no_answers":    ("نہیں", "n", "no"),
        "colours": {
            "red": "سرخ", "orange": "نارنجی", "yellow": "پیلا",
            "green": "سبز", "teal": "سبزی مائل نیلا", "blue": "نیلا",
            "purple": "جامنی", "pink": "گلابی", "white": "سفید",
            "light grey": "ہلکا سرمئی", "grey": "سرمئی", "dark grey": "گہرا سرمئی",
            "black": "کالا", "brown": "بھورا", "beige": "بیج",
        },
    },
}


# ---------------------------------------------------------------------------
# Named colour palette for clothing hint (BGR order to match OpenCV)
# ---------------------------------------------------------------------------
_COLOUR_PALETTE = {
    "red":        (  0,   0, 180),
    "orange":     (  0, 100, 220),
    "yellow":     (  0, 210, 230),
    "green":      ( 30, 140,  30),
    "teal":       (120, 140,  30),
    "blue":       (180,  60,  20),
    "purple":     (140,  30, 120),
    "pink":       (160,  80, 200),
    "white":      (220, 220, 220),
    "light grey": (180, 180, 180),
    "grey":       (120, 120, 120),
    "dark grey":  ( 70,  70,  70),
    "black":      ( 20,  20,  20),
    "brown":      ( 30,  60, 100),
    "beige":      (180, 200, 210),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect faces in an image and draw bounding rectangles."
    )
    parser.add_argument("image_path", help="Path to the input image file.")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path for the output image. Defaults to <input>_detected.<ext>.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Scale factor for the sliding window (default: 1.1).",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=8,
        help="Minimum neighbours required per detection (default: 8).",
    )
    parser.add_argument(
        "--min-size",
        default="80x80",
        help="Minimum face size as WxH, e.g. 80x80 (default: 80x80).",
    )
    return parser.parse_args()


def build_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_detected{input_path.suffix}")


# ---------------------------------------------------------------------------
# Language selection
# ---------------------------------------------------------------------------

def select_language() -> dict:
    """Show the language menu and return the chosen strings dict."""
    en = TRANSLATIONS["en"]
    while True:
        code = input(en["select_prompt"]).strip().lower()
        if code in TRANSLATIONS:
            return TRANSLATIONS[code]
        print(en["invalid_lang"])


# ---------------------------------------------------------------------------
# User input helpers  (all prompts supplied by caller via strings dict)
# ---------------------------------------------------------------------------

def ask_yes_no(prompt: str, strings: dict) -> bool:
    """Prompt for yes/no using language-aware accepted answers."""
    yes = strings["yes_answers"]
    no  = strings["no_answers"]
    while True:
        answer = input(prompt).strip().lower()
        if answer in yes:
            return True
        if answer in no:
            return False
        print(strings["invalid_yn"])


def ask_blur_style(strings: dict) -> str:
    """Prompt for blur style; style keywords are always Latin letters."""
    while True:
        answer = input(strings["style_prompt"]).strip().lower()
        if answer in ("square", "s"):
            return "square"
        if answer in ("oval", "o"):
            return "oval"
        if answer in ("strong", "t"):
            return "strong"
        if answer in ("silhouette", "l"):
            return "silhouette"
        if answer in ("emoji", "e"):
            return "emoji"
        print(strings["style_invalid"])


# ---------------------------------------------------------------------------
# Clothing colour hint
# ---------------------------------------------------------------------------

def _nearest_colour_name(bgr: tuple[int, int, int]) -> str:
    """Return the name of the closest colour in _COLOUR_PALETTE to the given BGR value."""
    b, g, r = bgr
    best_name, best_dist = "unknown", float("inf")
    for name, (pb, pg, pr) in _COLOUR_PALETTE.items():
        dist = (b - pb) ** 2 + (g - pg) ** 2 + (r - pr) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def dominant_clothing_colour(image, x: int, y: int, w: int, h: int,
                              strings: dict) -> str:
    """Sample strip below face bbox; return colour name in selected language."""
    img_h, img_w = image.shape[:2]
    strip_top = y + h
    strip_bottom = min(strip_top + h // 2, img_h)
    strip_left = max(x, 0)
    strip_right = min(x + w, img_w)

    if strip_top >= img_h or strip_left >= strip_right:
        return strings["colours"].get("unknown", "unknown")

    strip = image[strip_top:strip_bottom, strip_left:strip_right]
    mean_bgr = strip.mean(axis=(0, 1))
    en_name = _nearest_colour_name((int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])))
    return strings["colours"].get(en_name, en_name)


# ---------------------------------------------------------------------------
# Blur implementations
# ---------------------------------------------------------------------------

def blur_face_square(image, x: int, y: int, w: int, h: int) -> None:
    """Pixelate the face ROI: shrink to 10×10 then scale back up."""
    face_roi = image[y:y + h, x:x + w]
    small = cv2.resize(face_roi, (10, 10), interpolation=cv2.INTER_LINEAR)
    image[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _ellipse_mask(h: int, w: int) -> np.ndarray:
    """Return a (h, w) uint8 mask that is 255 inside a centred ellipse, 0 outside."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        mask,
        center=(w // 2, h // 2),
        axes=(w // 2, h // 2),
        angle=0, startAngle=0, endAngle=360,
        color=255, thickness=-1,
    )
    return mask


def blur_face_oval(image, x: int, y: int, w: int, h: int) -> None:
    """Pixelate only inside an ellipse fitted to the face bbox."""
    face_roi = image[y:y + h, x:x + w].copy()
    small = cv2.resize(face_roi, (10, 10), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_3ch = _ellipse_mask(h, w)[:, :, np.newaxis]
    image[y:y + h, x:x + w] = np.where(mask_3ch == 255, pixelated, face_roi)


def blur_face_strong(image, x: int, y: int, w: int, h: int) -> None:
    """Apply a heavy soft Gaussian blur (artistic, non-pixelated)."""
    face_roi = image[y:y + h, x:x + w]
    # Three passes of a large-kernel Gaussian give a smooth, painterly effect.
    blurred = face_roi
    for _ in range(3):
        blurred = cv2.GaussianBlur(blurred, (99, 99), sigmaX=30)
    image[y:y + h, x:x + w] = blurred


def blur_face_silhouette(image, x: int, y: int, w: int, h: int) -> None:
    """Replace the face with a solid dark oval silhouette."""
    mask = _ellipse_mask(h, w)
    # Paint dark charcoal (#1a1a1a in BGR) wherever the mask is filled.
    image[y:y + h, x:x + w][mask == 255] = (26, 26, 26)


_EMOJI_FONT_PATH = "C:/Windows/Fonts/seguiemj.ttf"

# Available emoji choices: key = short name, value = Unicode character.
EMOJI_OPTIONS: dict[str, str] = {
    "happy":    "\U0001f60a",  # 😊
    "laughing": "\U0001f602",  # 😂
    "crying":   "\U0001f622",  # 😢
    "cool":     "\U0001f60e",  # 😎
    "shy":      "\U0001f648",  # 🙈
}


def ask_emoji_choice() -> tuple[str, str]:
    """Ask the user which emoji to use; returns (name, char).

    Emoji glyphs are universal so no translation is needed for the options
    themselves — only the prompt wrapper is kept in English.
    """
    options = "  " + "\n  ".join(
        f"{i+1}. {char} {name}" for i, (name, char) in enumerate(EMOJI_OPTIONS.items())
    )
    keys = list(EMOJI_OPTIONS.keys())
    while True:
        print(f"  Pick an emoji:\n{options}")
        answer = input("  Your choice (1-5 or name): ").strip().lower()
        # Accept a number
        if answer.isdigit() and 1 <= int(answer) <= len(keys):
            name = keys[int(answer) - 1]
            return name, EMOJI_OPTIONS[name]
        # Accept the name directly
        if answer in EMOJI_OPTIONS:
            return answer, EMOJI_OPTIONS[answer]
        print(f"  Please enter a number 1-{len(keys)} or one of: {', '.join(keys)}.")


def _render_emoji_to_fit(target_w: int, target_h: int, char: str) -> np.ndarray:
    """Render `char` and scale it to exactly (target_h, target_w, 4) RGBA.

    Pillow's emoji glyph metrics are unreliable for sizing — instead we render
    onto a large canvas, crop the actual ink bounding box, then resize to fit.
    """
    canvas_size = max(target_w, target_h) * 4
    font_size = int(canvas_size * 0.7)
    font = ImageFont.truetype(_EMOJI_FONT_PATH, font_size)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    cx, cy = canvas_size // 2, canvas_size // 2
    draw.text((cx, cy), char, font=font, anchor="mm", embedded_color=True)

    arr = np.array(canvas)
    alpha_mask = arr[:, :, 3] > 0
    rows = np.where(np.any(alpha_mask, axis=1))[0]
    cols = np.where(np.any(alpha_mask, axis=0))[0]
    if len(rows) == 0:
        return np.zeros((target_h, target_w, 4), dtype=np.uint8)

    cropped = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    pil_cropped = Image.fromarray(cropped, mode="RGBA")
    pil_resized = pil_cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.array(pil_resized)


def blur_face_emoji(image, x: int, y: int, w: int, h: int, char: str) -> None:
    """Overlay the given emoji character sized to exactly cover the face region."""
    emoji_arr = _render_emoji_to_fit(w, h, char)        # (h, w, 4) RGBA
    alpha = emoji_arr[:, :, 3:4].astype(float) / 255
    fg_bgr = emoji_arr[:, :, :3].astype(float)[:, :, ::-1]  # RGB → BGR
    bg = image[y:y + h, x:x + w].astype(float)
    image[y:y + h, x:x + w] = (fg_bgr * alpha + bg * (1 - alpha)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Analysis helpers  (AI label · scene · safe-content score)
# ---------------------------------------------------------------------------

# Known AI-tool signatures in EXIF Software / ProcessingSoftware / Comment tags.
_AI_SIGNATURES = {
    "midjourney", "stable diffusion", "dall-e", "dalle", "firefly",
    "adobe firefly", "generative fill", "ideogram", "leonardo.ai",
    "nightcafe", "runwayml", "ai generated", "ai-generated",
    "diffusion", "gpt", "imagen",
}


def detect_ai_content(image_path: str) -> bool:
    """Return True if EXIF metadata suggests AI-generated or AI-edited content.

    Checks the Software, ProcessingSoftware, ImageDescription, and UserComment
    EXIF tags for known AI-tool keywords.  Also returns True if the file is a
    PNG with an 'AI' keyword in its tEXt chunks (common for Midjourney exports).
    """
    try:
        pil_img = Image.open(image_path)
        # --- JPEG / TIFF EXIF ---
        exif_data = pil_img._getexif() if hasattr(pil_img, "_getexif") else None
        if exif_data:
            # Tag IDs: 305=Software, 315=Artist, 270=ImageDescription, 37510=UserComment
            check_tags = [305, 315, 270, 37510]
            for tag_id in check_tags:
                val = exif_data.get(tag_id, "")
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="ignore")
                if any(sig in val.lower() for sig in _AI_SIGNATURES):
                    return True
        # --- PNG tEXt chunks ---
        if pil_img.format == "PNG":
            info = pil_img.info or {}
            combined = " ".join(str(v) for v in info.values()).lower()
            if any(sig in combined for sig in _AI_SIGNATURES):
                return True
    except Exception:
        pass
    return False


def stamp_ai_watermark(image, strings: dict) -> None:
    """Burn a visible red 'AI-GENERATED CONTENT' banner onto the bottom of the image."""
    h, w = image.shape[:2]
    font_scale = max(0.6, w / 2000)
    thickness = max(1, int(font_scale * 2))
    label = strings["ai_watermark"]
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    # Semi-transparent dark bar at the bottom
    bar_h = th + baseline + 16
    overlay = image.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    # Centred red text
    tx = (w - tw) // 2
    ty = h - baseline - 8
    cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX,
                font_scale, (0, 0, 220), thickness, cv2.LINE_AA)


def classify_scene(image) -> str:
    """Return 'indoor' or 'outdoor' using a colour + edge heuristic.

    Outdoor signal: top third of image has significant sky-blue saturation
    OR high edge density (trees, buildings, horizon).
    Indoor signal: muted tones, low saturation overall, low edge density.
    """
    h, w = image.shape[:2]
    top_third = image[:h // 3, :]

    # Convert top strip to HSV to measure sky-blue saturation
    hsv = cv2.cvtColor(top_third, cv2.COLOR_BGR2HSV)
    # Sky blue: hue 90-130 (OpenCV 0-180), saturation > 50, value > 80
    sky_mask = cv2.inRange(hsv, (90, 50, 80), (130, 255, 255))
    sky_ratio = sky_mask.sum() / (sky_mask.size * 255)

    # Edge density in top third via Canny
    gray_top = cv2.cvtColor(top_third, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_top, 50, 150)
    edge_density = edges.sum() / (edges.size * 255)

    # Overall image brightness & saturation
    hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv_full[:, :, 1].mean())

    # Decision: sky present OR high edges in top OR good overall saturation → outdoor
    if sky_ratio > 0.08 or (edge_density > 0.06 and mean_sat > 40):
        return "outdoor"
    return "indoor"


def safe_content_score(tally: dict) -> int:
    """Return a 0-100 compliance score.

    100 = all faces consented or properly blurred.
    Each skipped (unblurred, no-consent) face deducts proportionally.
    Returns 100 when no faces were detected.
    """
    total = tally["consented"] + sum(tally["blurred"].values()) + tally["skipped"]
    if total == 0:
        return 100
    compliant = tally["consented"] + sum(tally["blurred"].values())
    return round(100 * compliant / total)


def badge_label(score: int) -> str:
    """Return the gamification badge string for a given score."""
    if score == 100:
        return "\U0001f3c6 Perfect Privacy Badge — fully compliant!"
    if score >= 80:
        return "\u2705 Safe Creator Badge — great job!"
    if score >= 50:
        return "\u26a0\ufe0f Needs Review — some faces unaddressed."
    return "\U0001f534 High Risk — not safe to share yet."


# ---------------------------------------------------------------------------
# Consent flow
# ---------------------------------------------------------------------------

def print_summary(tally: dict, strings: dict, ai_flag: bool, scene: str) -> None:
    """Print the full end-of-run summary in the selected language."""
    blurred_total = sum(tally["blurred"].values())
    blur_breakdown = ", ".join(
        f"{count} {style}" for style, count in sorted(tally["blurred"].items()) if count
    ) or "0"
    print(strings["summary"].format(
        total=tally["consented"] + blurred_total + tally["skipped"],
        consented=tally["consented"],
        blurred=blurred_total,
        blur_breakdown=blur_breakdown,
        skipped=tally["skipped"],
    ))
    # Scene
    print(strings["scene_indoor"] if scene == "indoor" else strings["scene_outdoor"])
    # AI label
    if ai_flag:
        print(strings["ai_label"])
    # Safe content score
    score = safe_content_score(tally)
    if score >= 80:
        risk = strings["score_safe"]
    elif score >= 50:
        risk = strings["score_medium"]
    else:
        risk = strings["score_high"]
    print(strings["score_line"].format(score=score, risk=risk))
    print(badge_label(score))


def run_consent_flow(image, faces, strings: dict) -> tuple[dict, list[dict]]:
    """
    For each detected face:
      1. Sample the clothing colour below the bbox (translated) and include in prompt.
      2. Ask for consent in the selected language.
         - yes  → draw a numbered green bounding rectangle.
         - no   → ask whether to blur, then ask blur style.

    Returns (tally, face_details) where tally = {consented, blurred, skipped}
    and face_details is a list of per-face dicts for the PDF report.
    """
    tally: dict = {"consented": 0, "blurred": {}, "skipped": 0}
    face_details: list[dict] = []
    total = len(faces)
    for i, (x, y, w, h) in enumerate(faces, start=1):
        face_label = (
            strings["face_label"].format(i=i, total=total)
            if total > 1 else strings["face_label_1"]
        )
        colour = dominant_clothing_colour(image, x, y, w, h, strings)
        print()
        has_consent = ask_yes_no(
            strings["consent_q"].format(face_label=face_label, colour=colour),
            strings,
        )

        if has_consent:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"face {i}" if total > 1 else "face"
            cv2.putText(image, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1, cv2.LINE_AA)
            print(strings["consent_yes"].format(i=i))
            tally["consented"] += 1
            face_details.append({"index": i, "outcome": "consented", "style": "-", "colour": colour})
        else:
            should_blur = ask_yes_no(strings["blur_q"], strings)
            if should_blur:
                style = ask_blur_style(strings)
                if style == "oval":
                    blur_face_oval(image, x, y, w, h)
                elif style == "strong":
                    blur_face_strong(image, x, y, w, h)
                elif style == "silhouette":
                    blur_face_silhouette(image, x, y, w, h)
                elif style == "emoji":
                    emoji_name, emoji_char = ask_emoji_choice()
                    blur_face_emoji(image, x, y, w, h, emoji_char)
                    style = f"emoji:{emoji_name}"  # enrich for tally/PDF
                else:
                    blur_face_square(image, x, y, w, h)
                print(strings["blur_done"].format(i=i, style=style))
                tally["blurred"][style] = tally["blurred"].get(style, 0) + 1
                face_details.append({"index": i, "outcome": "blurred", "style": style, "colour": colour})
            else:
                print(strings["blur_no"].format(i=i))
                tally["skipped"] += 1
                face_details.append({"index": i, "outcome": "skipped", "style": "-", "colour": colour})
    return tally, face_details


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def nms_faces(faces, iou_threshold: float = 0.3):
    """Remove duplicate face boxes using non-maximum suppression.

    Sorts by area (largest first) and suppresses any box whose IoU with an
    already-kept box exceeds iou_threshold.  This handles Haar cascade's
    tendency to emit a large box and one or more smaller overlapping boxes
    for the same physical face.
    """
    if len(faces) == 0:
        return faces

    # Sort largest area first so we always keep the more confident big box.
    boxes = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    kept = []

    for box in boxes:
        x1, y1, w1, h1 = box
        area1 = w1 * h1
        discard = False
        for kx, ky, kw, kh in kept:
            # Intersection rectangle
            ix1 = max(x1, kx)
            iy1 = max(y1, ky)
            ix2 = min(x1 + w1, kx + kw)
            iy2 = min(y1 + h1, ky + kh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = w1 * h1 + kw * kh - inter
            # Suppress if IoU exceeds threshold OR if this box is mostly
            # contained within a larger kept box (handles small duplicates
            # that Haar cascade emits inside a larger detection of the same face).
            contained = inter / area1 if area1 > 0 else 0
            if (union > 0 and inter / union > iou_threshold) or contained > 0.6:
                discard = True
                break
        if not discard:
            kept.append(box)

    return kept


# ---------------------------------------------------------------------------
# Detection entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PDF compliance report
# ---------------------------------------------------------------------------

def generate_pdf_report(
    image_path: str,
    output_image_path: str,
    tally: dict,
    ai_flag: bool,
    scene: str,
    strings: dict,
    face_details: list[dict],
) -> str:
    """Generate a PDF compliance report and return its file path.

    face_details: list of dicts with keys 'index', 'outcome', 'style', 'colour'
                  (populated by run_consent_flow).
    """
    report_path = str(Path(output_image_path).with_suffix("")) + "_report.pdf"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lang = strings.get("lang_name", "English")

    # --- Derived values ---
    blurred_total = sum(tally["blurred"].values())
    skipped       = tally["skipped"]
    consented     = tally["consented"]
    total_faces   = consented + blurred_total + skipped
    score         = safe_content_score(tally)
    if score >= 80:
        risk_label = strings["score_safe"]
        risk_colour = (34, 139, 34)      # forest green
    elif score >= 50:
        risk_label = strings["score_medium"]
        risk_colour = (210, 120, 0)      # amber
    else:
        risk_label = strings["score_high"]
        risk_colour = (180, 30, 30)      # red

    # --- PDF setup ---
    pdf = FPDF()
    pdf.add_font("Segoe",  "",  _PDF_FONT_REGULAR)
    pdf.add_font("Segoe",  "B", _PDF_FONT_BOLD)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin  # usable width

    def h1(text: str) -> None:
        pdf.set_font("Segoe", "B", 18)
        pdf.set_text_color(20, 40, 80)
        pdf.cell(W, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    def h2(text: str) -> None:
        pdf.set_font("Segoe", "B", 13)
        pdf.set_text_color(40, 80, 140)
        pdf.set_fill_color(235, 241, 250)
        pdf.cell(W, 8, f"  {text}", fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    def body(text: str, indent: int = 0) -> None:
        pdf.set_font("Segoe", "", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(indent)
        pdf.multi_cell(W - indent, 7, text,
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def rule() -> None:
        pdf.set_draw_color(200, 200, 200)
        pdf.line(pdf.l_margin, pdf.get_y(),
                 pdf.l_margin + W, pdf.get_y())
        pdf.ln(3)

    # ── Header bar ──────────────────────────────────────────────────────────
    pdf.set_fill_color(20, 40, 80)
    pdf.rect(pdf.l_margin, pdf.get_y(), W, 14, style="F")
    pdf.set_font("Segoe", "B", 13)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(pdf.get_y() + 2)
    pdf.cell(W, 10, "  ConsentAI  |  Compliance Report",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(40, 40, 40)
    pdf.ln(4)

    # ── Title ────────────────────────────────────────────────────────────────
    h1("Content Compliance Report")
    pdf.set_font("Segoe", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(W, 6, f"Generated: {now}   |   Language: {lang}   |   Image: {Path(image_path).name}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    rule()

    # ── Section 1: Image info ────────────────────────────────────────────────
    h2("1. Image Information")
    img_cv = cv2.imread(image_path)
    if img_cv is not None:
        ih, iw = img_cv.shape[:2]
        body(f"File:       {Path(image_path).name}")
        body(f"Dimensions: {iw} x {ih} px")
    scene_str = strings["scene_indoor"] if scene == "indoor" else strings["scene_outdoor"]
    body(f"Scene:      {scene_str}")
    ai_str = strings["ai_label"] if ai_flag else "No AI-generated content markers detected."
    body(f"AI content: {ai_str}")
    pdf.ln(3)

    # ── Section 2: Face processing summary ───────────────────────────────────
    h2("2. Face Processing Summary")
    body(f"Total faces detected:  {total_faces}")
    body(f"  Consented (marked):  {consented}")
    body(f"  Blurred:             {blurred_total}")
    if tally["blurred"]:
        breakdown = ", ".join(f"{c} {s}" for s, c in sorted(tally["blurred"].items()) if c)
        body(f"    Breakdown:         {breakdown}", indent=4)
    body(f"  Left as-is (skipped): {skipped}")
    pdf.ln(2)

    # Per-face table
    if face_details:
        col_w = [10, 30, 35, 50]   # index | outcome | style | colour
        headers = ["#", "Outcome", "Blur style", "Clothing colour"]
        pdf.set_font("Segoe", "B", 10)
        pdf.set_fill_color(215, 225, 245)
        pdf.set_text_color(20, 40, 80)
        for cw, hdr in zip(col_w, headers):
            pdf.cell(cw, 7, hdr, border=1, fill=True)
        pdf.ln()
        pdf.set_font("Segoe", "", 10)
        pdf.set_text_color(40, 40, 40)
        for row in face_details:
            pdf.set_fill_color(248, 250, 255)
            pdf.cell(col_w[0], 7, str(row["index"]), border=1, fill=True)
            pdf.cell(col_w[1], 7, row["outcome"],    border=1, fill=True)
            pdf.cell(col_w[2], 7, row["style"],      border=1, fill=True)
            pdf.cell(col_w[3], 7, row["colour"],     border=1, fill=True)
            pdf.ln()
    pdf.ln(3)

    # ── Section 3: Safe Content Score ────────────────────────────────────────
    h2("3. Safe Content Score")
    # Large score display
    pdf.set_font("Segoe", "B", 38)
    pdf.set_text_color(*risk_colour)
    pdf.cell(W, 14, f"{score}/100", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Segoe", "B", 13)
    pdf.cell(W, 8, risk_label.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Segoe", "", 12)
    # Strip leading emoji glyph (outside Segoe UI's range) — keep the label text.
    badge = badge_label(score)
    badge_text = badge.split(" ", 1)[1] if badge[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" else badge
    pdf.cell(W, 8, badge_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Segoe", "", 10)
    pdf.multi_cell(W, 6,
        "Score formula: 100 x (consented + blurred) / total faces detected.\n"
        "100 = fully compliant. Each face left without consent or blur reduces the score.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # ── Section 4: Recommendations ───────────────────────────────────────────
    h2("4. Recommendations")
    recs: list[str] = []
    if skipped > 0:
        recs.append(
            f"{skipped} face(s) were left without consent or blur. "
            "Review these before publishing to avoid privacy violations."
        )
    if ai_flag:
        recs.append(
            "AI-generated content was detected. Under the EU AI Act, "
            "clearly label this content as AI-generated in all publications."
        )
    if score < 80:
        recs.append(
            "Safe Content Score is below 80. Obtain explicit consent or "
            "apply privacy blur to all unprocessed faces before sharing."
        )
    if not recs:
        recs.append(
            "All faces have been consented or properly anonymised. "
            "This image appears compliant for sharing."
        )
    for rec in recs:
        pdf.set_font("Segoe", "B", 11)
        pdf.set_text_color(20, 40, 80)
        pdf.cell(6, 7, "\u2022")          # bullet
        pdf.set_font("Segoe", "", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(W - 6, 7, rec,
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
    pdf.ln(2)

    # ── Footer ───────────────────────────────────────────────────────────────
    rule()
    pdf.set_font("Segoe", "", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(W, 6,
             f"ConsentAI  |  Report generated {now}  |  Output: {Path(output_image_path).name}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(report_path)
    return report_path


def detect_faces(image_path: str, output_path: str, scale_factor: float,
                 min_neighbors: int, min_size: tuple[int, int],
                 strings: dict) -> int:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not load image '{image_path}'.", file=sys.stderr)
        sys.exit(1)

    # --- Analysis: AI flag and scene classification ---
    ai_flag = detect_ai_content(image_path)
    scene   = classify_scene(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade_path = str(Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml")
    classifier = cv2.CascadeClassifier(cascade_path)
    if classifier.empty():
        print("Error: failed to load Haar cascade classifier.", file=sys.stderr)
        sys.exit(1)

    faces = classifier.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    raw = list(faces) if hasattr(faces, "__iter__") else []
    faces = nms_faces(raw)
    face_count = len(faces)

    face_details: list[dict] = []
    if face_count == 0:
        tally: dict = {"consented": 0, "blurred": {}, "skipped": 0}
        print(strings["none_detected"])
    else:
        print(strings["detected_n"].format(n=face_count))
        tally, face_details = run_consent_flow(image, faces, strings)

    # Stamp watermark on image before saving if AI-generated
    if ai_flag:
        stamp_ai_watermark(image, strings)

    print_summary(tally, strings, ai_flag, scene)
    cv2.imwrite(output_path, image)

    report_path = generate_pdf_report(
        image_path, output_path, tally, ai_flag, scene, strings, face_details
    )
    print(f"  Report saved to: {report_path}")

    return face_count


def main() -> None:
    strings = select_language()
    args = parse_args()

    input_path = Path(args.image_path)
    if not input_path.is_file():
        print(f"Error: '{input_path}' is not a file or does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        min_w, min_h = (int(v) for v in args.min_size.lower().split("x"))
    except ValueError:
        print("Error: --min-size must be in WxH format, e.g. 30x30.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(build_output_path(input_path))

    detect_faces(
        str(input_path),
        output_path,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        min_size=(min_w, min_h),
        strings=strings,
    )

    print(strings["saved"].format(path=output_path))


if __name__ == "__main__":
    main()
