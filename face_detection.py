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
import os
import threading
import time

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
        "rtl": False,
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
        "consent_duration_q": "  How long is consent valid? (seconds, default {default}): ",
        "consent_duration_invalid": "  Please enter a positive whole number of seconds.",
        "consent_expired":   "  Consent for face {i} has already expired — face will be blurred.",
        "consent_timer_label": "{secs}s",
        "expiry_watching":   "\nWatching for consent expiry — image will update automatically.",
        "expiry_fired":      "  [timer] Consent for face {i} expired — face blurred and image re-saved.",
        "expiry_all_done":   "  [timer] All consent timers have fired. Final image saved.",
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
        "ai_none":       "No AI-generated content markers detected.",
        "scene_indoor":  "Scene: indoors.",
        "scene_outdoor": "Scene: outdoors.",
        "score_line":    "Safe Content Score: {score}/100 ({risk}).",
        "score_safe":    "safe",
        "score_medium":  "medium risk",
        "score_high":    "high risk",
        "pdf_title":          "Content Compliance Report",
        "pdf_generated":      "Generated: {now}   |   Language: {lang}   |   Image: {name}",
        "pdf_s1_title":       "1. Image Information",
        "pdf_file":           "File",
        "pdf_dimensions":     "Dimensions",
        "pdf_scene":          "Scene",
        "pdf_ai_content":     "AI content",
        "pdf_s2_title":       "2. Face Processing Summary",
        "pdf_total":          "Total faces detected",
        "pdf_consented":      "Consented (marked)",
        "pdf_blurred":        "Blurred",
        "pdf_breakdown":      "Breakdown",
        "pdf_skipped":        "Left as-is (skipped)",
        "pdf_col_idx":        "#",
        "pdf_col_outcome":    "Outcome",
        "pdf_col_style":      "Blur style",
        "pdf_col_colour":     "Clothing colour",
        "pdf_col_consent":    "Consent (secs)",
        "pdf_expired":        "expired",
        "pdf_s3_title":       "3. Safe Content Score",
        "pdf_score_formula":  "Score formula: 100 x (consented + blurred) / total faces detected.\n"
                              "100 = fully compliant. Each face left without consent or blur reduces the score.",
        "pdf_s4_title":       "4. Recommendations",
        "pdf_rec_skipped":    "{n} face(s) were left without consent or blur. "
                              "Review these before publishing to avoid privacy violations.",
        "pdf_rec_ai":         "AI-generated content was detected. Under the EU AI Act, "
                              "clearly label this content as AI-generated in all publications.",
        "pdf_rec_score":      "Safe Content Score is below 80. Obtain explicit consent or "
                              "apply privacy blur to all unprocessed faces before sharing.",
        "pdf_rec_ok":         "All faces have been consented or properly anonymised. "
                              "This image appears compliant for sharing.",
        "pdf_footer":         "Compliact  |  Report generated {now}  |  Output: {name}",
        "save_error":    "  [error] Failed to write image to: {path}",
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
        "rtl": False,
        "select_prompt": "",  # only shown in English
        "invalid_lang":  "",
        "detected_n":    "{n} visage(s) détecté(s).",
        "none_detected": "Aucun visage détecté.",
        "face_label":    "Visage {i} sur {total}",
        "face_label_1":  "Une personne",
        "consent_q":     "{face_label} détecté — couleur vestimentaire : {colour}. "
                         "Avez-vous le consentement de cette personne pour partager cette photo ? (oui/non) : ",
        "consent_yes":   "  Parfait — visage {i} marqué.",
        "consent_duration_q": "  Combien de temps dure le consentement ? (secondes, défaut {default}) : ",
        "consent_duration_invalid": "  Veuillez entrer un nombre entier positif de secondes.",
        "consent_expired":   "  Le consentement pour le visage {i} a expiré — le visage sera flouté.",
        "consent_timer_label": "{secs}s",
        "expiry_watching":   "\nSurveillance des expirations de consentement — l'image se mettra à jour automatiquement.",
        "expiry_fired":      "  [minuterie] Le consentement pour le visage {i} a expiré — flouté et image ré-enregistrée.",
        "expiry_all_done":   "  [minuterie] Toutes les minuteries ont expiré. Image finale enregistrée.",
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
        "ai_none":       "Aucun marqueur de contenu généré par IA détecté.",
        "scene_indoor":  "Scène : intérieur.",
        "scene_outdoor": "Scène : extérieur.",
        "score_line":    "Score de contenu sûr : {score}/100 ({risk}).",
        "score_safe":    "sûr",
        "score_medium":  "risque moyen",
        "score_high":    "risque élevé",
        "pdf_title":          "Rapport de conformité du contenu",
        "pdf_generated":      "Généré : {now}   |   Langue : {lang}   |   Image : {name}",
        "pdf_s1_title":       "1. Informations sur l'image",
        "pdf_file":           "Fichier",
        "pdf_dimensions":     "Dimensions",
        "pdf_scene":          "Scène",
        "pdf_ai_content":     "Contenu IA",
        "pdf_s2_title":       "2. Résumé du traitement des visages",
        "pdf_total":          "Total de visages détectés",
        "pdf_consented":      "Consentis (marqués)",
        "pdf_blurred":        "Floutés",
        "pdf_breakdown":      "Détail",
        "pdf_skipped":        "Laissés tels quels",
        "pdf_col_idx":        "#",
        "pdf_col_outcome":    "Résultat",
        "pdf_col_style":      "Style de flou",
        "pdf_col_colour":     "Couleur vêtement",
        "pdf_col_consent":    "Consentement (sec)",
        "pdf_expired":        "expiré",
        "pdf_s3_title":       "3. Score de contenu sûr",
        "pdf_score_formula":  "Formule : 100 x (consentis + floutés) / total visages détectés.\n"
                              "100 = totalement conforme. Chaque visage non traité réduit le score.",
        "pdf_s4_title":       "4. Recommandations",
        "pdf_rec_skipped":    "{n} visage(s) sans consentement ni floutage. "
                              "Examinez ces cas avant publication pour éviter les violations de confidentialité.",
        "pdf_rec_ai":         "Contenu généré par IA détecté. Conformément à la loi IA UE, "
                              "étiquetez clairement ce contenu comme généré par IA dans toutes les publications.",
        "pdf_rec_score":      "Le score est inférieur à 80. Obtenez le consentement explicite ou "
                              "appliquez un flou de confidentialité à tous les visages non traités.",
        "pdf_rec_ok":         "Tous les visages ont été consentis ou correctement anonymisés. "
                              "Cette image semble conforme pour publication.",
        "pdf_footer":         "Compliact  |  Rapport généré le {now}  |  Sortie : {name}",
        "save_error":    "  [erreur] Impossible d'écrire l'image dans : {path}",
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
        "rtl": False,
        "select_prompt": "",
        "invalid_lang":  "",
        "detected_n":    "Se detectaron {n} cara(s).",
        "none_detected": "No se detectaron caras.",
        "face_label":    "Cara {i} de {total}",
        "face_label_1":  "Una persona",
        "consent_q":     "{face_label} detectada — color de ropa: {colour}. "
                         "¿Tienes el consentimiento de esta persona para compartir la foto? (sí/no): ",
        "consent_yes":   "  Entendido — cara {i} marcada.",
        "consent_duration_q": "  ¿Cuánto dura el consentimiento? (segundos, por defecto {default}): ",
        "consent_duration_invalid": "  Por favor ingresa un número entero positivo de segundos.",
        "consent_expired":   "  El consentimiento para la cara {i} ha expirado — se difuminará.",
        "consent_timer_label": "{secs}s",
        "expiry_watching":   "\nEsperando expiración de consentimientos — la imagen se actualizará automáticamente.",
        "expiry_fired":      "  [temporizador] El consentimiento de la cara {i} expiró — difuminada y guardada.",
        "expiry_all_done":   "  [temporizador] Todos los temporizadores han expirado. Imagen final guardada.",
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
        "ai_none":       "No se detectaron marcadores de contenido generado por IA.",
        "scene_indoor":  "Escena: interior.",
        "scene_outdoor": "Escena: exterior.",
        "score_line":    "Puntuación de contenido seguro: {score}/100 ({risk}).",
        "score_safe":    "seguro",
        "score_medium":  "riesgo medio",
        "score_high":    "riesgo alto",
        "pdf_title":          "Informe de cumplimiento de contenido",
        "pdf_generated":      "Generado: {now}   |   Idioma: {lang}   |   Imagen: {name}",
        "pdf_s1_title":       "1. Información de la imagen",
        "pdf_file":           "Archivo",
        "pdf_dimensions":     "Dimensiones",
        "pdf_scene":          "Escena",
        "pdf_ai_content":     "Contenido IA",
        "pdf_s2_title":       "2. Resumen del procesamiento de caras",
        "pdf_total":          "Total de caras detectadas",
        "pdf_consented":      "Con consentimiento (marcadas)",
        "pdf_blurred":        "Difuminadas",
        "pdf_breakdown":      "Desglose",
        "pdf_skipped":        "Sin cambios",
        "pdf_col_idx":        "#",
        "pdf_col_outcome":    "Resultado",
        "pdf_col_style":      "Estilo de difuminado",
        "pdf_col_colour":     "Color de ropa",
        "pdf_col_consent":    "Consentimiento (seg)",
        "pdf_expired":        "expirado",
        "pdf_s3_title":       "3. Puntuación de contenido seguro",
        "pdf_score_formula":  "Fórmula: 100 x (consentidos + difuminados) / total caras detectadas.\n"
                              "100 = totalmente conforme. Cada cara sin tratar reduce la puntuación.",
        "pdf_s4_title":       "4. Recomendaciones",
        "pdf_rec_skipped":    "{n} cara(s) sin consentimiento ni difuminado. "
                              "Revisa estos casos antes de publicar para evitar violaciones de privacidad.",
        "pdf_rec_ai":         "Se detectó contenido generado por IA. Según la Ley IA UE, "
                              "etiqueta claramente este contenido como generado por IA en todas las publicaciones.",
        "pdf_rec_score":      "La puntuación es inferior a 80. Obtén consentimiento explícito o "
                              "aplica difuminado de privacidad a todas las caras no procesadas.",
        "pdf_rec_ok":         "Todas las caras han sido consentidas o correctamente anonimizadas. "
                              "Esta imagen parece conforme para publicar.",
        "pdf_footer":         "Compliact  |  Informe generado el {now}  |  Salida: {name}",
        "save_error":    "  [error] No se pudo escribir la imagen en: {path}",
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
        "rtl": True,
        "select_prompt": "",
        "invalid_lang":  "",
        "detected_n":    "تم اكتشاف {n} وجه/وجوه.",
        "none_detected": "لم يتم اكتشاف أي وجوه.",
        "face_label":    "الوجه {i} من {total}",
        "face_label_1":  "شخص",
        "consent_q":     "{face_label} — لون الملابس يبدو {colour}. "
                         "هل لديك موافقة هذا الشخص لمشاركة الصورة؟ (نعم/لا): ",
        "consent_yes":   "  تمام — تم تحديد الوجه {i}.",
        "consent_duration_q": "  كم تدوم الموافقة؟ (ثانية، الافتراضي {default}): ",
        "consent_duration_invalid": "  الرجاء إدخال عدد صحيح موجب من الثوانٍ.",
        "consent_expired":   "  انتهت صلاحية موافقة الوجه {i} — سيتم تمويهه.",
        "consent_timer_label": "{secs}ث",
        "expiry_watching":   "\nمراقبة انتهاء صلاحية الموافقات — سيتم تحديث الصورة تلقائياً.",
        "expiry_fired":      "  [مؤقت] انتهت موافقة الوجه {i} — تم التمويه وإعادة الحفظ.",
        "expiry_all_done":   "  [مؤقت] انتهت جميع المؤقتات. تم حفظ الصورة النهائية.",
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
        "ai_none":       "لم يتم اكتشاف أي علامات على محتوى مُنشأ بالذكاء الاصطناعي.",
        "scene_indoor":  "المشهد: داخلي.",
        "scene_outdoor": "المشهد: خارجي.",
        "score_line":    "نقاط المحتوى الآمن: {score}/100 ({risk}).",
        "score_safe":    "آمن",
        "score_medium":  "خطر متوسط",
        "score_high":    "خطر مرتفع",
        "pdf_title":          "تقرير الامتثال للمحتوى",
        "pdf_generated":      "تاريخ الإنشاء: {now}   |   اللغة: {lang}   |   الصورة: {name}",
        "pdf_s1_title":       "1. معلومات الصورة",
        "pdf_file":           "الملف",
        "pdf_dimensions":     "الأبعاد",
        "pdf_scene":          "المشهد",
        "pdf_ai_content":     "محتوى الذكاء الاصطناعي",
        "pdf_s2_title":       "2. ملخص معالجة الوجوه",
        "pdf_total":          "إجمالي الوجوه المكتشفة",
        "pdf_consented":      "موافق عليها (مُعلَّمة)",
        "pdf_blurred":        "مموَّهة",
        "pdf_breakdown":      "التفصيل",
        "pdf_skipped":        "متروكة كما هي",
        "pdf_col_idx":        "#",
        "pdf_col_outcome":    "النتيجة",
        "pdf_col_style":      "أسلوب التمويه",
        "pdf_col_colour":     "لون الملابس",
        "pdf_col_consent":    "الموافقة (ثانية)",
        "pdf_expired":        "منتهية",
        "pdf_s3_title":       "3. نقاط المحتوى الآمن",
        "pdf_score_formula":  "الصيغة: 100 × (الموافق عليها + المموَّهة) / إجمالي الوجوه المكتشفة.\n"
                              "100 = امتثال كامل. كل وجه غير معالج يخفض النقاط.",
        "pdf_s4_title":       "4. التوصيات",
        "pdf_rec_skipped":    "تُرك {n} وجه/وجوه دون موافقة أو تمويه. "
                              "راجع هذه الحالات قبل النشر لتجنب انتهاكات الخصوصية.",
        "pdf_rec_ai":         "تم اكتشاف محتوى مُنشأ بالذكاء الاصطناعي. وفق قانون الذكاء الاصطناعي الأوروبي، "
                              "صنِّف هذا المحتوى بوضوح كمحتوى ذكاء اصطناعي في جميع المنشورات.",
        "pdf_rec_score":      "النقاط أقل من 80. احصل على موافقة صريحة أو "
                              "طبِّق تمويه الخصوصية على جميع الوجوه غير المعالجة.",
        "pdf_rec_ok":         "تمت الموافقة على جميع الوجوه أو إخفاء هويتها بشكل صحيح. "
                              "تبدو هذه الصورة متوافقة للنشر.",
        "pdf_footer":         "Compliact  |  تم إنشاء التقرير: {now}  |  المخرج: {name}",
        "save_error":    "  [خطأ] تعذّر كتابة الصورة إلى: {path}",
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
        "rtl": False,
        "select_prompt": "",
        "invalid_lang":  "",
        "detected_n":    "{n} rosto(s) detectado(s).",
        "none_detected": "Nenhum rosto detectado.",
        "face_label":    "Rosto {i} de {total}",
        "face_label_1":  "Uma pessoa",
        "consent_q":     "{face_label} detectado — cor da roupa: {colour}. "
                         "Você tem o consentimento dessa pessoa para compartilhar a foto? (sim/não): ",
        "consent_yes":   "  Certo — rosto {i} marcado.",
        "consent_duration_q": "  Por quanto tempo o consentimento é válido? (segundos, padrão {default}): ",
        "consent_duration_invalid": "  Por favor insira um número inteiro positivo de segundos.",
        "consent_expired":   "  O consentimento para o rosto {i} expirou — o rosto será desfocado.",
        "consent_timer_label": "{secs}s",
        "expiry_watching":   "\nAguardando expiração dos consentimentos — a imagem será atualizada automaticamente.",
        "expiry_fired":      "  [temporizador] O consentimento do rosto {i} expirou — desfocado e imagem re-salva.",
        "expiry_all_done":   "  [temporizador] Todos os temporizadores dispararam. Imagem final salva.",
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
        "ai_none":       "Nenhum marcador de conteúdo gerado por IA detectado.",
        "scene_indoor":  "Cena: interior.",
        "scene_outdoor": "Cena: exterior.",
        "score_line":    "Pontuação de conteúdo seguro: {score}/100 ({risk}).",
        "score_safe":    "seguro",
        "score_medium":  "risco médio",
        "score_high":    "risco alto",
        "pdf_title":          "Relatório de conformidade de conteúdo",
        "pdf_generated":      "Gerado: {now}   |   Idioma: {lang}   |   Imagem: {name}",
        "pdf_s1_title":       "1. Informações da imagem",
        "pdf_file":           "Arquivo",
        "pdf_dimensions":     "Dimensões",
        "pdf_scene":          "Cena",
        "pdf_ai_content":     "Conteúdo IA",
        "pdf_s2_title":       "2. Resumo do processamento de rostos",
        "pdf_total":          "Total de rostos detectados",
        "pdf_consented":      "Com consentimento (marcados)",
        "pdf_blurred":        "Desfocados",
        "pdf_breakdown":      "Detalhamento",
        "pdf_skipped":        "Sem alteração",
        "pdf_col_idx":        "#",
        "pdf_col_outcome":    "Resultado",
        "pdf_col_style":      "Estilo de desfoque",
        "pdf_col_colour":     "Cor da roupa",
        "pdf_col_consent":    "Consentimento (seg)",
        "pdf_expired":        "expirado",
        "pdf_s3_title":       "3. Pontuação de conteúdo seguro",
        "pdf_score_formula":  "Fórmula: 100 x (consentidos + desfocados) / total de rostos detectados.\n"
                              "100 = totalmente conforme. Cada rosto não tratado reduz a pontuação.",
        "pdf_s4_title":       "4. Recomendações",
        "pdf_rec_skipped":    "{n} rosto(s) sem consentimento ou desfoque. "
                              "Revise estes casos antes de publicar para evitar violações de privacidade.",
        "pdf_rec_ai":         "Conteúdo gerado por IA detectado. Conforme a Lei IA UE, "
                              "rotule claramente este conteúdo como gerado por IA em todas as publicações.",
        "pdf_rec_score":      "A pontuação está abaixo de 80. Obtenha consentimento explícito ou "
                              "aplique desfoque de privacidade a todos os rostos não processados.",
        "pdf_rec_ok":         "Todos os rostos foram consentidos ou devidamente anonimizados. "
                              "Esta imagem parece estar em conformidade para publicação.",
        "pdf_footer":         "Compliact  |  Relatório gerado em {now}  |  Saída: {name}",
        "save_error":    "  [erro] Falha ao gravar imagem em: {path}",
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
        "rtl": False,
        "select_prompt": "",
        "invalid_lang":  "",
        "detected_n":    "{n} Gesicht(er) erkannt.",
        "none_detected": "Keine Gesichter erkannt.",
        "face_label":    "Gesicht {i} von {total}",
        "face_label_1":  "Eine Person",
        "consent_q":     "{face_label} erkannt — Kleidungsfarbe: {colour}. "
                         "Liegt das Einverständnis dieser Person vor, das Foto zu teilen? (ja/nein): ",
        "consent_yes":   "  Alles klar — Gesicht {i} markiert.",
        "consent_duration_q": "  Wie lange gilt das Einverständnis? (Sekunden, Standard {default}): ",
        "consent_duration_invalid": "  Bitte eine positive ganze Zahl für Sekunden eingeben.",
        "consent_expired":   "  Das Einverständnis für Gesicht {i} ist abgelaufen — Gesicht wird unkenntlich gemacht.",
        "consent_timer_label": "{secs}s",
        "expiry_watching":   "\nWarte auf Ablauf der Einverständnisse — Bild wird automatisch aktualisiert.",
        "expiry_fired":      "  [Timer] Einverständnis für Gesicht {i} abgelaufen — unkenntlich gemacht und Bild neu gespeichert.",
        "expiry_all_done":   "  [Timer] Alle Timer abgelaufen. Finales Bild gespeichert.",
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
        "ai_none":       "Keine Marker für KI-generierte Inhalte erkannt.",
        "scene_indoor":  "Szene: drinnen.",
        "scene_outdoor": "Szene: draußen.",
        "score_line":    "Inhaltssicherheitsbewertung: {score}/100 ({risk}).",
        "score_safe":    "sicher",
        "score_medium":  "mittleres Risiko",
        "score_high":    "hohes Risiko",
        "pdf_title":          "Inhalts-Compliance-Bericht",
        "pdf_generated":      "Erstellt: {now}   |   Sprache: {lang}   |   Bild: {name}",
        "pdf_s1_title":       "1. Bildinformationen",
        "pdf_file":           "Datei",
        "pdf_dimensions":     "Abmessungen",
        "pdf_scene":          "Szene",
        "pdf_ai_content":     "KI-Inhalt",
        "pdf_s2_title":       "2. Zusammenfassung der Gesichtsverarbeitung",
        "pdf_total":          "Erkannte Gesichter gesamt",
        "pdf_consented":      "Mit Einverständnis (markiert)",
        "pdf_blurred":        "Unkenntlich gemacht",
        "pdf_breakdown":      "Aufschlüsselung",
        "pdf_skipped":        "Unverändert",
        "pdf_col_idx":        "#",
        "pdf_col_outcome":    "Ergebnis",
        "pdf_col_style":      "Unkenntlichungs-Stil",
        "pdf_col_colour":     "Kleidungsfarbe",
        "pdf_col_consent":    "Einverständnis (Sek.)",
        "pdf_expired":        "abgelaufen",
        "pdf_s3_title":       "3. Inhaltssicherheitsbewertung",
        "pdf_score_formula":  "Formel: 100 x (einverstanden + unkenntlich) / erkannte Gesichter gesamt.\n"
                              "100 = vollständig konform. Jedes unbehandelte Gesicht senkt die Bewertung.",
        "pdf_s4_title":       "4. Empfehlungen",
        "pdf_rec_skipped":    "{n} Gesicht(er) ohne Einverständnis oder Unkenntlichmachung hinterlassen. "
                              "Bitte vor der Veröffentlichung prüfen, um Datenschutzverletzungen zu vermeiden.",
        "pdf_rec_ai":         "KI-generierter Inhalt erkannt. Gemäß EU-KI-Gesetz "
                              "diesen Inhalt in allen Veröffentlichungen klar als KI-generiert kennzeichnen.",
        "pdf_rec_score":      "Bewertung unter 80. Ausdrückliches Einverständnis einholen oder "
                              "Datenschutz-Unkenntlichmachung auf alle unbehandelten Gesichter anwenden.",
        "pdf_rec_ok":         "Alle Gesichter wurden zugestimmt oder ordnungsgemäß anonymisiert. "
                              "Dieses Bild scheint konform zum Teilen.",
        "pdf_footer":         "Compliact  |  Bericht erstellt am {now}  |  Ausgabe: {name}",
        "save_error":    "  [Fehler] Bild konnte nicht geschrieben werden nach: {path}",
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
        "rtl": True,
        "select_prompt": "",
        "invalid_lang":  "",
        "detected_n":    "{n} چہرہ/چہرے پائے گئے۔",
        "none_detected": "کوئی چہرہ نہیں ملا۔",
        "face_label":    "چہرہ {i} از {total}",
        "face_label_1":  "ایک شخص",
        "consent_q":     "{face_label} ملا — لباس کا رنگ {colour} لگتا ہے۔ "
                         "کیا آپ کے پاس اس شخص کی اجازت ہے کہ یہ تصویر شیئر کریں؟ (ہاں/نہیں): ",
        "consent_yes":   "  بالکل — چہرہ {i} نشان زد کر دیا گیا۔",
        "consent_duration_q": "  رضامندی کتنے وقت کے لیے درست ہے؟ (سیکنڈ، پہلے سے طے {default}): ",
        "consent_duration_invalid": "  براہ کرم سیکنڈ کی ایک مثبت پوری تعداد درج کریں۔",
        "consent_expired":   "  چہرہ {i} کی رضامندی ختم ہو گئی — چہرہ دھندلا کر دیا جائے گا۔",
        "consent_timer_label": "{secs}s",
        "expiry_watching":   "\nرضامندیوں کی میعاد ختم ہونے کا انتظار — تصویر خود بخود اپڈیٹ ہو گی۔",
        "expiry_fired":      "  [ٹائمر] چہرہ {i} کی رضامندی ختم — دھندلا کر کے دوبارہ محفوظ کیا گیا۔",
        "expiry_all_done":   "  [ٹائمر] تمام ٹائمر ختم ہو گئے۔ آخری تصویر محفوظ۔",
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
        "ai_none":       "AI سے تیار کردہ مواد کی کوئی علامت نہیں ملی۔",
        "scene_indoor":  "منظر: اندرونی۔",
        "scene_outdoor": "منظر: بیرونی۔",
        "score_line":    "محفوظ مواد اسکور: {score}/100 ({risk})۔",
        "score_safe":    "محفوظ",
        "score_medium":  "درمیانہ خطرہ",
        "score_high":    "زیادہ خطرہ",
        "pdf_title":          "مواد کی تعمیل رپورٹ",
        "pdf_generated":      "تیار کردہ: {now}   |   زبان: {lang}   |   تصویر: {name}",
        "pdf_s1_title":       "1. تصویر کی معلومات",
        "pdf_file":           "فائل",
        "pdf_dimensions":     "سائز",
        "pdf_scene":          "منظر",
        "pdf_ai_content":     "AI مواد",
        "pdf_s2_title":       "2. چہروں کی پروسیسنگ کا خلاصہ",
        "pdf_total":          "کل شناخت شدہ چہرے",
        "pdf_consented":      "رضامندی والے (نشان زد)",
        "pdf_blurred":        "دھندلے کیے گئے",
        "pdf_breakdown":      "تفصیل",
        "pdf_skipped":        "بغیر تبدیلی",
        "pdf_col_idx":        "#",
        "pdf_col_outcome":    "نتیجہ",
        "pdf_col_style":      "دھندلاپن کا انداز",
        "pdf_col_colour":     "لباس کا رنگ",
        "pdf_col_consent":    "رضامندی (سیکنڈ)",
        "pdf_expired":        "ختم",
        "pdf_s3_title":       "3. محفوظ مواد اسکور",
        "pdf_score_formula":  "فارمولہ: 100 × (رضامندی + دھندلے) ÷ کل چہرے۔\n"
                              "100 = مکمل تعمیل۔ ہر غیر عملدرآمد چہرہ اسکور کم کرتا ہے۔",
        "pdf_s4_title":       "4. سفارشات",
        "pdf_rec_skipped":    "{n} چہرہ/چہرے بغیر رضامندی یا دھندلاپن کے چھوڑے گئے۔ "
                              "اشاعت سے پہلے ان کا جائزہ لیں تاکہ رازداری کی خلاف ورزی سے بچا جا سکے۔",
        "pdf_rec_ai":         "AI سے تیار کردہ مواد ملا۔ EU AI Act کے تحت "
                              "اسے تمام اشاعتوں میں AI سے تیار کردہ کے طور پر واضح طور پر لیبل کریں۔",
        "pdf_rec_score":      "اسکور 80 سے کم ہے۔ واضح رضامندی حاصل کریں یا "
                              "تمام غیر عملدرآمد چہروں پر رازداری کا دھندلاپن لگائیں۔",
        "pdf_rec_ok":         "تمام چہروں کی رضامندی حاصل کر لی گئی یا انہیں مناسب طریقے سے گمنام کر دیا گیا۔ "
                              "یہ تصویر اشاعت کے لیے تعمیل کے مطابق دکھتی ہے۔",
        "pdf_footer":         "Compliact  |  رپورٹ تیار کردہ {now}  |  آؤٹ پٹ: {name}",
        "save_error":    "  [خرابی] تصویر یہاں محفوظ نہیں ہو سکی: {path}",
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
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return input_path.with_name(f"{input_path.stem}_detected_{ts}{input_path.suffix}")


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
# Consent duration helpers
# ---------------------------------------------------------------------------

#: Default consent window in seconds shown as the placeholder in the prompt.
_DEFAULT_CONSENT_SECONDS: int = 30


def ask_consent_duration(strings: dict) -> int:
    """Ask how many seconds consent is valid for.

    Returns the number of seconds entered by the user (minimum 1).
    Pressing Enter with no value uses *_DEFAULT_CONSENT_SECONDS*.
    """
    prompt = strings["consent_duration_q"].format(default=_DEFAULT_CONSENT_SECONDS)
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return _DEFAULT_CONSENT_SECONDS
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print(strings["consent_duration_invalid"])


def draw_consent_timer(
    image,
    x: int,
    y: int,
    w: int,
    h: int,
    seconds_remaining: int,
    strings: dict,
) -> None:
    """Draw a countdown badge in the top-right corner of the face bounding box.

    The badge shows the remaining consent time (e.g. "30s").  Colour shifts
    from green → amber → red as the clock winds down relative to the original
    grant duration (uses absolute value for colouring):
      > 60 s  green
      > 10 s  amber
      ≤ 10 s  red
    """
    label = strings["consent_timer_label"].format(secs=seconds_remaining)

    # Colour: green → amber → red
    if seconds_remaining > 60:
        bg_colour  = (34, 139, 34)    # forest green  (BGR)
        txt_colour = (255, 255, 255)
    elif seconds_remaining > 10:
        bg_colour  = (0, 165, 255)    # amber/orange  (BGR)
        txt_colour = (255, 255, 255)
    else:
        bg_colour  = (30, 30, 200)    # red           (BGR)
        txt_colour = (255, 255, 255)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(0.6, w / 120))
    thickness  = 1

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    pad    = 4
    bx1    = x + w - tw - pad * 2      # badge left
    by1    = y - th - pad * 2 - 2      # badge top  (just above the box)
    bx2    = x + w                      # badge right
    by2    = y                          # badge bottom

    # Keep badge inside image boundaries
    img_h, img_w = image.shape[:2]
    if by1 < 0:
        by1 = y
        by2 = y + th + pad * 2
    bx1 = max(0, bx1)

    # Filled rectangle background
    cv2.rectangle(image, (bx1, by1), (bx2, by2), bg_colour, cv2.FILLED)
    # Text centred in badge
    tx = bx1 + pad
    ty = by2 - pad - baseline
    cv2.putText(image, label, (tx, ty), font, font_scale,
                txt_colour, thickness, cv2.LINE_AA)


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

# def blur_face_square(image, x: int, y: int, w: int, h: int) -> None:
#     """Pixelate the face ROI: shrink to a small grid then scale back up."""
#     if w <= 0 or h <= 0:
#         return
#     # Use a grid size that's always at least 2 pixels but scales with face size
#     grid = max(2, min(w, h) // 8)
#     face_roi = image[y:y + h, x:x + w]
#     small = cv2.resize(face_roi, (grid, grid), interpolation=cv2.INTER_LINEAR)
#     image[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def blur_face_square(image, x: int, y: int, w: int, h: int) -> None:
    """Pixelate the face ROI: shrink to a small grid then scale back up."""
    if w <= 0 or h <= 0:
        return

    # 1. Extract the ROI (NumPy automatically handles edge truncation)
    face_roi = image[max(0, y):y + h, max(0, x):x + w]
    
    # 2. Get the actual dimensions of the extracted slice
    actual_h, actual_w = face_roi.shape[:2]
    if actual_h == 0 or actual_w == 0:
        return

    # 3. Calculate grid size based on actual width/height
    grid: int = max(2, min(actual_w, actual_h) // 16)

    # 4. Downsample to pixelate
    small = cv2.resize(face_roi, (grid, grid), interpolation=cv2.INTER_LINEAR)
    
    # 5. Upsample back to the EXACT dimensions of the slice to prevent crash
    image[max(0, y):y + h, max(0, x):x + w] = cv2.resize(
        small, (actual_w, actual_h), interpolation=cv2.INTER_NEAREST
    )


def _ellipse_mask(h: int, w: int) -> np.ndarray:
    """Return a (h, w) uint8 mask that is 255 inside a centred ellipse, 0 outside."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        mask,
        center=(w // 2, h // 2),
        axes=(w // 2, h // 2),
        angle=0, startAngle=0, endAngle=360,
        color=(255,255,255), thickness=-1,
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
    if w <= 0 or h <= 0:
        return
    face_roi = image[y:y + h, x:x + w]
    # Kernel must be odd and smaller than the ROI dimensions
    ksize = min(w if w % 2 == 1 else w - 1, h if h % 2 == 1 else h - 1, 99)
    ksize = max(ksize, 3)
    blurred = face_roi
    for _ in range(3):
        blurred = cv2.GaussianBlur(blurred, (ksize, ksize), sigmaX=30)
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
        exif_data = pil_img.getexif() if hasattr(pil_img, "_getexif") else None
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
    sky_mask = cv2.inRange(hsv, np.array([90, 50, 80]), np.array([130, 255, 255]))
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
         - yes  → ask for consent duration (seconds), draw green bbox + countdown badge.
                   If the supplied duration is 0 the consent is treated as already expired
                   and the face is blurred automatically.
         - no   → ask whether to blur, then ask blur style.

    Dynamic-blur / consent expiry
    ──────────────────────────────
    When consent is granted the user specifies how many seconds it is valid for.
    The remaining time is stamped as a coloured badge (green / amber / red) next to
    the face on the output image.  If the user enters 0 seconds the consent is
    considered expired and the face is blurred immediately, mirroring the behaviour
    of a live countdown reaching zero.

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
            # ── Dynamic blur: ask how long consent is valid ──────────────────
            consent_secs = ask_consent_duration(strings)
            granted_at   = time.time()          # wall-clock moment consent was given

            if consent_secs <= 0:
                # Consent has already expired — auto-blur
                print(strings["consent_expired"].format(i=i))
                blur_face_square(image, x, y, w, h)
                tally["blurred"]["square"] = tally["blurred"].get("square", 0) + 1
                face_details.append({
                    "index": i, "outcome": "blurred", "style": "square",
                    "colour": colour, "consent_secs": 0, "_granted_at": granted_at,
                    "bbox": (x, y, w, h),
                })
            else:
                # Consent is active — mark face and stamp countdown badge
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"face {i}" if total > 1 else "face"
                cv2.putText(image, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0), 1, cv2.LINE_AA)
                draw_consent_timer(image, x, y, w, h, consent_secs, strings)
                print(strings["consent_yes"].format(i=i))
                tally["consented"] += 1
                face_details.append({
                    "index": i, "outcome": "consented", "style": "-",
                    "colour": colour, "consent_secs": consent_secs, "_granted_at": granted_at,
                    "bbox": (x, y, w, h),
                })
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
                face_details.append({
                    "index": i, "outcome": "blurred", "style": style,
                    "colour": colour, "consent_secs": None,
                    "bbox": (x, y, w, h),
                })
            else:
                print(strings["blur_no"].format(i=i))
                tally["skipped"] += 1
                face_details.append({
                    "index": i, "outcome": "skipped", "style": "-",
                    "colour": colour, "consent_secs": None,
                    "bbox": (x, y, w, h),
                })
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
        risk_colour = (34, 139, 34)
    elif score >= 50:
        risk_label = strings["score_medium"]
        risk_colour = (210, 120, 0)
    else:
        risk_label = strings["score_high"]
        risk_colour = (180, 30, 30)

    # --- PDF setup ---
    pdf = FPDF()
    pdf.add_font("Segoe",  "",  _PDF_FONT_REGULAR)
    pdf.add_font("Segoe",  "B", _PDF_FONT_BOLD)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin   # usable width (190 mm on A4)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def h1(text: str) -> None:
        pdf.set_font("Segoe", "B", 18)
        pdf.set_text_color(20, 40, 80)
        pdf.multi_cell(W, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    def h2(text: str) -> None:
        pdf.set_font("Segoe", "B", 13)
        pdf.set_text_color(40, 80, 140)
        pdf.set_fill_color(235, 241, 250)
        pdf.multi_cell(W, 8, f"  {text}", fill=True,
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    def body(text: str) -> None:
        pdf.set_font("Segoe", "", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(W, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def rule() -> None:
        pdf.set_draw_color(200, 200, 200)
        pdf.line(pdf.l_margin, pdf.get_y(),
                 pdf.l_margin + W, pdf.get_y())
        pdf.ln(3)

    # Two-column label/value layout — LW (label) + VW (value) = W exactly.
    LW = 48
    VW = W - LW

    def row2(label: str, value: str) -> None:
        """Bold label (LW mm) on the left, regular value (VW mm) on the right.
        Both use multi_cell so long text wraps inside its column; nothing clips.
        """
        row_y = pdf.get_y()

        pdf.set_font("Segoe", "B", 11)
        pdf.set_text_color(80, 80, 80)
        pdf.set_xy(pdf.l_margin, row_y)
        pdf.multi_cell(LW, 7, label, align="L",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        label_bottom = pdf.get_y()

        pdf.set_font("Segoe", "", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.set_xy(pdf.l_margin + LW, row_y)
        pdf.multi_cell(VW, 7, value, align="L",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        value_bottom = pdf.get_y()

        pdf.set_y(max(label_bottom, value_bottom))

    # ── Header bar ──────────────────────────────────────────────────────────
    pdf.set_fill_color(20, 40, 80)
    pdf.rect(pdf.l_margin, pdf.get_y(), W, 14, style="F")
    pdf.set_font("Segoe", "B", 13)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(pdf.get_y() + 2)
    pdf.multi_cell(W, 10, "  Compliact  |  " + strings["pdf_title"],
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(40, 40, 40)
    pdf.ln(2)

    # ── Title ────────────────────────────────────────────────────────────────
    h1(strings["pdf_title"])
    pdf.set_font("Segoe", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(W, 6,
                   strings["pdf_generated"].format(
                       now=now, lang=lang, name=Path(image_path).name),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    rule()

    # ── Section 1: Image info ────────────────────────────────────────────────
    h2(strings["pdf_s1_title"])
    img_cv = cv2.imread(image_path)
    if img_cv is not None:
        ih, iw = img_cv.shape[:2]
        row2(strings["pdf_file"],       Path(image_path).name)
        row2(strings["pdf_dimensions"], f"{iw} x {ih} px")
    scene_str = strings["scene_indoor"] if scene == "indoor" else strings["scene_outdoor"]
    # Strip leading ⚠ — Segoe UI in fpdf2 cannot render that glyph
    ai_str = strings["ai_label"] if ai_flag else strings["ai_none"]
    ai_str = ai_str.lstrip("⚠ ")
    row2(strings["pdf_scene"],      scene_str)
    row2(strings["pdf_ai_content"], ai_str)
    pdf.ln(2)

    # ── Section 2: Face processing summary ───────────────────────────────────
    h2(strings["pdf_s2_title"])
    row2(strings["pdf_total"],     str(total_faces))
    row2(strings["pdf_consented"], str(consented))
    row2(strings["pdf_blurred"],   str(blurred_total))
    if tally["blurred"]:
        breakdown = ", ".join(f"{c} {s}" for s, c in sorted(tally["blurred"].items()) if c)
        row2(strings["pdf_breakdown"], breakdown)
    row2(strings["pdf_skipped"], str(skipped))
    pdf.ln(2)

    # ── Per-face table — columns sum exactly to W ────────────────────────────
    if face_details:
        c0 = 8; c1 = 30; c2 = 32; c4 = 28; c3 = W - c0 - c1 - c2 - c4
        col_w   = [c0, c1, c2, c3, c4]
        headers = [
            strings["pdf_col_idx"],
            strings["pdf_col_outcome"],
            strings["pdf_col_style"],
            strings["pdf_col_colour"],
            strings["pdf_col_consent"],
        ]
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
            secs = row.get("consent_secs")
            if secs is None:
                consent_cell = "-"
            elif secs == 0:
                consent_cell = strings["pdf_expired"]
            else:
                consent_cell = f"{secs}s"
            pdf.cell(col_w[0], 7, str(row["index"]), border=1, fill=True)
            pdf.cell(col_w[1], 7, row["outcome"],    border=1, fill=True)
            pdf.cell(col_w[2], 7, row["style"],      border=1, fill=True)
            pdf.cell(col_w[3], 7, row["colour"],     border=1, fill=True)
            pdf.cell(col_w[4], 7, consent_cell,      border=1, fill=True)
            pdf.ln()
    pdf.ln(2)

    # ── Section 3: Safe Content Score ────────────────────────────────────────
    h2(strings["pdf_s3_title"])
    pdf.set_font("Segoe", "B", 28)
    pdf.set_text_color(*risk_colour)
    pdf.multi_cell(W, 10, f"{score}/100", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Segoe", "B", 13)
    pdf.multi_cell(W, 7, risk_label.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Segoe", "", 11)
    # Strip leading emoji glyph (outside Segoe UI's range)
    badge = badge_label(score)
    badge_text = badge.split(" ", 1)[1] if badge[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" else badge
    pdf.multi_cell(W, 7, badge_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Segoe", "", 10)
    pdf.multi_cell(W, 6, strings["pdf_score_formula"],
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # ── Section 4: Recommendations ───────────────────────────────────────────
    h2(strings["pdf_s4_title"])
    recs: list[str] = []
    if skipped > 0:
        recs.append(strings["pdf_rec_skipped"].format(n=skipped))
    if ai_flag:
        recs.append(strings["pdf_rec_ai"])
    if score < 80:
        recs.append(strings["pdf_rec_score"])
    if not recs:
        recs.append(strings["pdf_rec_ok"])
    for rec in recs:
        pdf.set_font("Segoe", "B", 11)
        pdf.set_text_color(20, 40, 80)
        pdf.cell(6, 7, "\u2022")
        pdf.set_font("Segoe", "", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(W - 6, 7, rec, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
    pdf.ln(1)

    # ── Footer ───────────────────────────────────────────────────────────────
    rule()
    pdf.set_font("Segoe", "", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(W, 6,
                   strings["pdf_footer"].format(
                       now=now, name=Path(output_image_path).name),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Write the PDF — if the file is open in a viewer (locked), fall back to a
    # timestamped name so the run is never lost.
    try:
        pdf.output(report_path)
    except PermissionError:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_path.replace("_report.pdf", f"_report_{ts}.pdf")
        pdf.output(report_path)
    return report_path


# ---------------------------------------------------------------------------
# Consent expiry watcher
# ---------------------------------------------------------------------------

def _render_expired_image(
    image_path: str,
    output_path: str,
    faces: list,
    face_details: list[dict],
    expired_indices: set[int],
    ai_flag: bool,
    strings: dict,
) -> None:
    """Re-render the output image with newly-expired faces blurred.

    Reloads the original source image and re-applies every face decision,
    replacing any consent badge that has now expired with a square blur.
    ``expired_indices`` is the *cumulative* set of face indices (1-based)
    whose consent has lapsed so far.
    """
    img = cv2.imread(image_path)
    if img is None:
        return

    for detail in face_details:
        i   = detail["index"]
        x, y, w, h = detail["bbox"]

        if detail["outcome"] == "consented":
            if i in expired_indices:
                # Timer has fired for this face — blur it now
                blur_face_square(img, x, y, w, h)
            else:
                # Still active — redraw box and live badge with remaining time
                secs_remaining = detail["consent_secs"] - int(time.time() - detail["_granted_at"])
                secs_remaining = max(1, secs_remaining)   # clamp: don't show 0 on a live face
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"face {i}" if len(faces) > 1 else "face"
                cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0), 1, cv2.LINE_AA)
                draw_consent_timer(img, x, y, w, h, secs_remaining, strings)

        elif detail["outcome"] == "blurred":
            style = detail["style"]
            if style == "oval":
                blur_face_oval(img, x, y, w, h)
            elif style == "strong":
                blur_face_strong(img, x, y, w, h)
            elif style == "silhouette":
                blur_face_silhouette(img, x, y, w, h)
            elif style.startswith("emoji:"):
                # emoji char stored separately — fall back to square on re-render
                blur_face_square(img, x, y, w, h)
            else:
                blur_face_square(img, x, y, w, h)

        # outcome == "skipped": leave the face untouched

    if ai_flag:
        stamp_ai_watermark(img, strings)

    if os.path.exists(output_path):
        os.remove(output_path)
    if not cv2.imwrite(output_path, img):
        print(strings["save_error"].format(path=output_path))


def expiry_watcher(
    image_path: str,
    output_path: str,
    faces: list,
    face_details: list[dict],
    ai_flag: bool,
    strings: dict,
) -> None:
    """Block until all consent timers have fired, re-rendering the image each time.

    Called in a background thread by detect_faces whenever at least one face
    has an active consent duration.  Each consented face gets its own timer;
    when it fires the image is re-saved with that face now blurred.
    """
    # Only track faces with an active (> 0) consent timer
    pending = [d for d in face_details if d.get("consent_secs") and d["consent_secs"] > 0]
    if not pending:
        return

    expired_indices: set[int] = set()
    # Sort by duration so shortest fires first
    pending.sort(key=lambda d: d["consent_secs"])

    for detail in pending:
        target_time = detail["_granted_at"] + detail["consent_secs"]
        wait = target_time - time.time()
        if wait > 0:
            time.sleep(wait)
        expired_indices.add(detail["index"])
        _render_expired_image(
            image_path, output_path, faces, face_details, expired_indices, ai_flag, strings
        )
        print(strings["expiry_fired"].format(i=detail["index"]))

    print(strings["expiry_all_done"])


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
    if os.path.exists(output_path):
        os.remove(output_path)
    cv2.imwrite(output_path, image)

    report_path = generate_pdf_report(
        image_path, output_path, tally, ai_flag, scene, strings, face_details
    )
    print(f"  Report saved to: {report_path}")

    # ── Launch expiry watcher if any consented faces have a live timer ────────
    active_timers = [d for d in face_details if d.get("consent_secs") and d["consent_secs"] > 0]
    if active_timers:
        print(strings["expiry_watching"])
        watcher = threading.Thread(
            target=expiry_watcher,
            args=(image_path, output_path, faces, face_details, ai_flag, strings),
            daemon=False,   # keep process alive until all timers fire
        )
        watcher.start()
        watcher.join()   # block main thread so the script stays alive

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
