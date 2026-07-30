# Compliact

An AI copilot that helps content creators and businesses check legal compliance before publishing photos or videos. Consent, privacy, and AI transparency, handled through a natural conversation instead of a static checklist.

---

## Problem Statement

Content creators publish photos and videos every day without an easy way to check whether they are legally exposed. Unclear consent from people appearing in the shot, missing GDPR related disclosures, or a lack of the mandatory "AI generated" transparency label that becomes required under the EU AI Act starting August 2, 2026. Existing tools are either manual, such as basic blur apps with no guidance, or built for enterprise compliance teams, not solo creators. This leaves individual creators and small businesses exposed to legal risk, and increasingly to the risk of having their own likeness misused in deepfake scams.

---

## Solution Description

Compliact is a conversational AI copilot that reviews a creator's photo before it gets posted. Instead of a silent scan or a static report, it holds a natural language conversation with the creator. It flags each person detected in the shot and asks whether consent was given, checks for AI generated content that requires disclosure, and evaluates the overall compliance risk. Based on the creator's answers, it directly executes the chosen action, such as blurring a face in one of five styles, or leaving it untouched if consent was given.

Consent in Compliact is not a one time checkbox. Creators can grant time limited consent with a visible countdown, and if that consent is not renewed before it lapses, Compliact automatically re blurs the face and updates both the image and the compliance report so the final record always matches the current state of the photo. A fully compliant result is rewarded with a Safe Content Score and a badge.

The experience is designed to feel like talking to a knowledgeable friend rather than filling out a compliance form, and works in seven languages so it can support creators internationally, not just in one market.

---

## Core User Flow

1. Creator provides a photo.
2. Compliact detects every face in the image.
3. For each face, it asks: "Do you have consent to share this person?"
4. If consent is given, the creator can set a time limit for that consent. If the timer expires before the creator returns to renew it, the face is automatically blurred and the image and report are updated.
5. If consent is not given, it asks whether to blur the face, and in which style: Square (pixelated), oval (face shaped soft blur), strong (heavy artistic blur), silhouette (solid shape), or emoji overlay, with a small choice of expressions.
6. It also checks the overall scene (indoor or outdoor) and whether the image appears AI generated, adding the required disclosure label automatically if so.
7. At the end, it delivers a friendly summary, a Safe Content Score out of 100, a compliance badge, and a downloadable PDF compliance report.

---

## AI Approach and Architecture

**Detection layer.** OpenCV Haar Cascade face detection, with a custom non maximum suppression step (using both IoU and a containment ratio check) to eliminate duplicate and false positive detections. A known weak point of lightweight Haar Cascade models on cluttered backgrounds.

**Conversational layer (IBM Bob).** Drives the natural language dialogue in the terminal, asking about consent and blur preferences in the user's selected language, with translations designed to sound natural rather than literal, covering all user facing messages including the final compliance badge and blur style names.

**Action layer.** Executes the selected outcome per face, drawing a live consent countdown badge, or applying the selected blur style using OpenCV image processing (Gaussian blur, elliptical masking, or emoji compositing via Pillow). A background watcher thread tracks every active consent timer and automatically re renders the image with the correct blur applied the moment a timer expires, using the original stored face coordinates so the correct face is always targeted.

**Compliance layer.** Scene classification (indoor or outdoor, using HSV sky color ratio and Canny edge density), AI generated content labeling for AI Act compliance, and a Safe Content Score calculated from the ratio of consented or blurred faces to total faces detected.

**Reporting layer.** Generates a structured PDF compliance report summarizing the analysis, score, and recommendations. The report automatically regenerates once all consent timers have fired, so it always reflects the final, current state of the image rather than the state at the moment of the initial upload.

---

## Selected Challenge Theme

July Challenge: Reimagine Creative Industries with AI.

---

## How IBM Bob Was Used

IBM Bob was the primary development tool used throughout the build, from the first working prototype to the final feature set. Specifically, Bob was used to:

1. Generate the initial face detection script and iteratively debug real compatibility issues, including an OpenCV major version break that removed CascadeClassifier from the default package, requiring a pinned dependency fix.
2. Diagnose and fix a duplicate detection bug where the same physical face was being detected twice by the Haar Cascade, by implementing a containment ratio check that standard IoU based deduplication missed.
3. Build the multilingual conversational flow across seven languages, including correct handling of native yes and no responses and Windows terminal encoding fixes for non Latin scripts.
4. Implement and refine all five blur styles, including debugging a visually incorrect blur where an oversized kernel produced a flat grey block instead of a real blur effect, and fixing emoji rendering and sizing.
5. Diagnose and fix a bug in the dynamic consent expiry feature where the wrong face was being blurred after a consent timer lapsed, tracing it to an index mismatch between the face detection list and the stored consent data, and resolving it by storing each face's coordinates directly with its consent record.
6. Add a write verification step so any failed image save is reported immediately instead of failing silently.
7. Add the AI generated content label, Safe Content Score, badge system, scene classification, and PDF report generation, including automatic report regeneration once consent timers expire.
8. Extend translation coverage across all seven supported languages so the compliance badge and blur style names display correctly in every language, not only in English.
9. Test the full pipeline end to end across languages, blur combinations, and consent expiry scenarios before freezing the code for submission.

---

## Known Limitations

Face detection uses a lightweight Haar Cascade model, which is fast and dependency light but less reliable on non frontal faces, extreme angles, or expressive poses. A future version would use a more robust DNN based face detector.

The automatic blur on consent expiry has been tested and confirmed working reliably on multi person photos with typically sized face detections. On a single close up portrait where one face fills most of the frame, the expiry blur does not currently apply correctly in the saved output, and this is being investigated further.

The PDF compliance report has a minor layout issue where some text can overflow its column in two sections. This does not affect the accuracy of the report content.

---

## Roadmap and Vision

Compliact's current prototype focuses on what could be reliably built and tested within the challenge timeframe. The broader product vision includes:

1. **Voice and deepfake detection.** Extending the same consent first philosophy to audio in video content.
2. **Logo and brand detection.** Helping creators and businesses avoid inadvertent copyright exposure.
3. **Enterprise Compliance Mode.** Stricter, configurable rules for businesses publishing content at scale, paired with the existing PDF report as an audit trail.
4. **Smart auto crop.** Automatically reframing an image around a consented subject instead of only blurring.
5. Any future work involving detection of sensitive categories, such as potential minors in content, would require rigorous fairness and bias testing before being presented as a reliable feature, given the well documented unreliability of age estimation models. This is treated as a long term research question, not a near term feature.

---

## Team

**Rameesha Munawar.** Lead and principal developer. Designed and built the core detection, consent, blur, scoring, and reporting pipeline, and diagnosed and fixed the consent expiry and translation bugs described above.

**Sepideh Mahmoodi.** Contributed the initial dynamic and expiring consent feature branch, set up the shared repository, completed the required IBM SkillsBuild learning, and identified key bugs including a right to left text alignment issue in the Arabic PDF output.

---

## Tech Stack

Python, OpenCV, Pillow, IBM Bob.
