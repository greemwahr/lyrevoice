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
- [ ] Quantitative evaluation — run `evaluate.py` to get FAD + speaker cosine similarity scores
- [ ] Qualitative evaluation — listen to generated audio samples, include examples
- [ ] Document how modifications affected quality — compare checkpoints across training runs, show impact of architecture/hyperparameter changes

## 5. Creative and Practical Applications
- [x] Prototype built — Gradio app (record 3 sentences, clone voice, speak any text)
- [ ] Test the Gradio app end-to-end with a real checkpoint
- [ ] Write up proposed applications — accessibility, content creation, etc.

## 6. Ethical Considerations
- [ ] Write discussion on: data privacy, deepfake/deceptive media risks, impact on voice actors/creative industries

## 7. Report and Presentation
- [ ] Project report — methodology, architecture decisions, experiments, results, conclusions
- [ ] Presentation — slides for class showcase with generated audio examples

## Submission Requirements
- [x] Complete source code with run instructions (RUNBOOK.md exists)
- [ ] Detailed project report
- [ ] Presentation slides
