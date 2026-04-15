"""
app/app.py

LyreVoice Gradio web application.

User flow:
  Step 1 -- Voice Enrollment
    User records 3 sample sentences into the microphone.
    Resemblyzer encodes these into a speaker embedding (voice fingerprint).

  Step 2 -- Text-to-Speech
    User types any text.
    The Generator synthesizes a mel-spectrogram conditioned on the text
    and the enrolled speaker embedding.
    HiFi-GAN converts the mel to a playable audio waveform.
    User hears their cloned voice speaking the typed text.

Usage:
  python app/app.py                        # Local development
  python app/app.py --share                # Temporary public URL (Gradio tunnel)
  python app/app.py --server_port 7860     # Custom port for server deployment
"""

import argparse
import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from models.generator import LyreVoiceGenerator
from models.speaker_encoder import SpeakerEncoder
from models.vocoder import Vocoder


# ─────────────────────────────────────────────
# Global model state (loaded once at startup)
# ─────────────────────────────────────────────

MODELS = {}
CONFIG = {}


def load_models(config_path: str, checkpoint_path: str) -> None:
    """Load all models into memory once at app startup."""
    global MODELS, CONFIG

    with open(config_path, "r") as f:
        CONFIG = yaml.safe_load(f)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading models on device: {device}")

    speaker_encoder = SpeakerEncoder(device=device)
    generator = LyreVoiceGenerator(CONFIG, device=device)
    vocoder = Vocoder(CONFIG, device=device)

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(ckpt["generator_state_dict"])
        print(f"Generator loaded from: {checkpoint_path}")
    else:
        print("No checkpoint provided -- using pretrained Tacotron2 weights only.")

    generator.eval()

    MODELS["speaker_encoder"] = speaker_encoder
    MODELS["generator"] = generator
    MODELS["vocoder"] = vocoder
    MODELS["device"] = device
    MODELS["sample_rate"] = CONFIG["audio"]["sample_rate"]

    print("All models loaded successfully.")


# ─────────────────────────────────────────────
# Core inference functions
# ─────────────────────────────────────────────

# Stores the enrolled speaker embedding between Gradio calls
_enrolled_embedding = {"embedding": None, "label": "No voice enrolled"}


def enroll_voice(audio_1, audio_2, audio_3) -> str:
    """
    Step 1: Encode 3 recorded audio clips into a speaker embedding.

    Each audio input is a (sample_rate, numpy_array) tuple from gr.Audio.
    Returns a status message shown to the user.
    """
    if audio_1 is None or audio_2 is None or audio_3 is None:
        return "Please record all 3 sentences before enrolling."

    try:
        sample_rate = MODELS["sample_rate"]
        speaker_encoder = MODELS["speaker_encoder"]

        # Gradio returns (sample_rate, numpy_array)
        audios = []
        for audio_input in [audio_1, audio_2, audio_3]:
            sr, audio_array = audio_input
            audio_float = audio_array.astype(np.float32)
            # Normalize to [-1, 1]
            if audio_float.max() > 1.0:
                audio_float = audio_float / 32768.0
            audios.append((audio_float, sr))

        # Encode each clip and average
        embeddings = []
        for audio_float, sr in audios:
            emb = speaker_encoder.encode_audio_array(audio_float, sr)
            embeddings.append(emb)

        mean_emb = torch.stack(embeddings).mean(dim=0)
        mean_emb = mean_emb / (mean_emb.norm() + 1e-8)

        _enrolled_embedding["embedding"] = mean_emb
        _enrolled_embedding["label"] = "Voice enrolled successfully"

        return "Voice enrolled! You can now generate speech in your voice."

    except Exception as e:
        return f"Enrollment failed: {str(e)}"


def generate_speech(text: str) -> tuple:
    """
    Step 2: Generate speech from text using the enrolled voice.

    Returns (sample_rate, audio_array) for gr.Audio output.
    """
    if not text or not text.strip():
        return None, "Please enter some text to synthesize."

    if _enrolled_embedding["embedding"] is None:
        return None, "Please enroll your voice first (Step 1)."

    try:
        generator = MODELS["generator"]
        vocoder = MODELS["vocoder"]
        sample_rate = MODELS["sample_rate"]
        spk_emb = _enrolled_embedding["embedding"]

        with torch.no_grad():
            mel = generator.infer(text.strip(), spk_emb)
            audio = vocoder.mel_to_audio(mel)

        # Gradio Audio component expects (sample_rate, numpy_array)
        return (sample_rate, audio), "Speech generated successfully."

    except Exception as e:
        return None, f"Generation failed: {str(e)}"


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

SAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood.",
]

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="LyreVoice -- Voice Cloning", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # LyreVoice
        ### AI-powered Voice Cloning
        Enroll your voice with 3 short recordings, then generate speech in your own voice from any text.
        """)

        # ── Step 1: Voice Enrollment ──────────────────────────────────────
        with gr.Tab("Step 1: Enroll Your Voice"):
            gr.Markdown("""
            Record yourself reading each sentence below.
            Speak clearly at a normal pace in a quiet environment.
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"**Sentence 1:** *{SAMPLE_SENTENCES[0]}*")
                    audio_1 = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Recording 1",
                    )
                with gr.Column():
                    gr.Markdown(f"**Sentence 2:** *{SAMPLE_SENTENCES[1]}*")
                    audio_2 = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Recording 2",
                    )
                with gr.Column():
                    gr.Markdown(f"**Sentence 3:** *{SAMPLE_SENTENCES[2]}*")
                    audio_3 = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Recording 3",
                    )

            enroll_btn = gr.Button("Enroll My Voice", variant="primary")
            enroll_status = gr.Textbox(label="Status", interactive=False)

            enroll_btn.click(
                fn=enroll_voice,
                inputs=[audio_1, audio_2, audio_3],
                outputs=enroll_status,
            )

        # ── Step 2: Text-to-Speech ────────────────────────────────────────
        with gr.Tab("Step 2: Generate Speech"):
            gr.Markdown("""
            Type any text below and click **Generate** to hear it spoken in your enrolled voice.
            """)

            text_input = gr.Textbox(
                label="Text to synthesize",
                placeholder="Type something here...",
                lines=3,
            )

            generate_btn = gr.Button("Generate Speech", variant="primary")

            with gr.Row():
                audio_output = gr.Audio(
                    label="Your cloned voice",
                    type="numpy",
                )
                gen_status = gr.Textbox(label="Status", interactive=False)

            generate_btn.click(
                fn=generate_speech,
                inputs=text_input,
                outputs=[audio_output, gen_status],
            )

            gr.Examples(
                examples=[
                    ["Welcome to LyreVoice, the AI-powered voice cloning app."],
                    ["Today is a beautiful day. I hope you are doing well."],
                    ["The future of voice technology is here, and it sounds just like you."],
                ],
                inputs=text_input,
            )

        # ── About ─────────────────────────────────────────────────────────
        with gr.Tab("About"):
            gr.Markdown("""
            ## How LyreVoice Works

            **LyreVoice** is a voice cloning system built with:
            - **Resemblyzer** -- extracts a unique voice fingerprint from your recordings
            - **Tacotron2 GAN** -- generates mel-spectrograms conditioned on your voice
            - **HiFi-GAN** -- converts mel-spectrograms to high-quality audio

            ### Architecture
            ```
            Your voice recordings
                    ↓
            [Resemblyzer] → 256-dim speaker embedding
                    ↓
            [Tacotron2 Generator] + text → mel-spectrogram
                    ↓
            [HiFi-GAN Vocoder] → audio waveform
                    ↓
            You hear your voice!
            ```

            ### Privacy
            All processing happens locally on this server.
            Your voice recordings are never stored or transmitted externally.

            ---
            *Built for LIAI 1009 - Deep Learning | Georgian College*
            """)

    return demo


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LyreVoice app")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--server_name", default="0.0.0.0",
                        help="0.0.0.0 to allow external connections")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio tunnel URL")
    args = parser.parse_args()

    load_models(args.config, args.checkpoint)

    demo = build_ui()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
