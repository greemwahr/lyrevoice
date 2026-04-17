# Assignment #2 Deliverables — LIAI 1009 Deep Learning

**Weight:** 40% of grade

---

## 1. Choice of Application
- [x] Domain chosen — voice cloning (text-to-speech with speaker identity)

## 2. GAN Model Development
- [x] GAN implemented (modified Tacotron2 generator + custom multi-scale discriminator)
- [x] Dataset selected and used (VCTK 110 speakers + LJSpeech)

## 3. Training the GAN
- [x] GAN trained (150 epochs on H100, G/D interplay documented)
- [x] Hyperparameter experimentation (2 runs: LR tuning, batch size changes, D LR reduction, lambda schedule)

## 4. Quality Assessment
- [ ] Quantitative evaluation — run `evaluate.py` on Colab to get FAD + speaker cosine similarity scores
- [ ] Qualitative evaluation — listen to generated audio samples, include examples
- [ ] Document how modifications affected quality — compare checkpoints across training runs
- [ ] Fill in evaluation results placeholders in Word report

## 5. Creative and Practical Applications
- [x] Prototype built — Gradio app (record 3 sentences, clone voice, speak any text)
- [ ] Test the Gradio app end-to-end with a real checkpoint
- [x] Write up proposed applications — in Word report (accessibility, content creation, preservation, education, entertainment)

## 6. Ethical Considerations
- [x] Written in Word report — data privacy, deepfake risks, creative industries impact, mitigations

## 7. Report and Presentation
- [x] Project report — `report/LyreVoice_Project_Report.docx` (Candara 12pt, all sections)
- [x] Presentation outline — `report/presentation_outline.md` (18 slides)
- [x] Architecture diagram — `report/lyrevoice_architecture.png` embedded in Word report Appendix B
- [ ] Presentation slides — convert outline to actual slides (PowerPoint/Google Slides)

## Submission Requirements
- [x] Complete source code with run instructions (RUNBOOK.md exists)
- [x] Detailed project report (Word doc in `report/`)
- [ ] Presentation slides (need actual .pptx or Google Slides)
