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
- [x] Quantitative evaluation — FAD + speaker cosine similarity for epochs 50, 100, 150 (best: epoch 100, FAD=63.64, sim=0.419)
- [x] Qualitative evaluation — observations on generated audio written in report Section 5.2
- [x] WandB training charts added to report (G/D loss, loss components, D scores, lambda schedule)
- [x] Document how modifications affected quality — impact table in Section 5.3, compromises/learnings in Section 4.5
- [x] Fill in evaluation results placeholders in Word report

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
