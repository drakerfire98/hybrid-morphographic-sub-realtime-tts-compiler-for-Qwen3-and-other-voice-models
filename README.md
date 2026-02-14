# ComfyUI-Qwen-TTS

English | [中文版](README_CN.md)

![Nodes Screenshot](example/qwen3-tts.png)

ComfyUI custom nodes for speech synthesis, voice cloning, and voice design, based on the open-source **Qwen3-TTS** project by the Alibaba Qwen team.

## 📋 Changelog

### 2026-01-23 - Dependency Compatibility & Mac Support
- **Fixed**: Resolved `transformers` version conflicts with `qwen-tts` dependency
- **Improvement**: Now supports local package import without requiring `pip install qwen-tts`
- **New**: Add MPS (Mac Apple Silicon) support for device detection
- **Note**: The official `qwen-tts` package requires `transformers==4.57.3`, which may conflict with other ComfyUI nodes. This version uses bundled local code to avoid dependency issues.

## Key Features

- 🎵 **Speech Synthesis**: High-quality text-to-speech conversion.
- 🎭 **Voice Cloning**: Zero-shot voice cloning from short reference audio.
- 🎨 **Voice Design**: Create custom voice characteristics based on natural language descriptions.
- 🚀 **Efficient Inference**: Supports both 12Hz and 25Hz speech tokenizer architectures.
- 🎯 **Multilingual**: Native support for 10 languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian).
- ⚡ **Integrated Loading**: No separate loader nodes required; model loading is managed on-demand with global caching.
- ⏱️ **Ultra-Low Latency**: Supports high-fidelity speech reconstruction with low-latency streaming.

## Nodes List

### 1. Qwen3-TTS Voice Design (`VoiceDesignNode`)
Generate unique voices based on text descriptions.
- **Inputs**:
  - `text`: Target text to synthesize.
  - `instruct`: Description of the voice (e.g., "A gentle female voice with a high pitch").
  - `model_choice`: Currently locked to **1.7B** for VoiceDesign features.
- **Capabilities**: Best for creating "imaginary" voices or specific character archetypes.

### 2. Qwen3-TTS Voice Clone (`VoiceCloneNode`)
Clone a voice from a reference audio clip.
- **Inputs**:
  - `ref_audio`: A short (5-15s) audio clip to clone.
  - `ref_text`: Text spoken in the `ref_audio` (helps improve quality).
  - `target_text`: The new text you want the cloned voice to say.
  - `model_choice`: Choose between **0.6B** (fast) or **1.7B** (high quality).

### 3. Qwen3-TTS Custom Voice (`CustomVoiceNode`)
Standard TTS using preset speakers.
- **Inputs**:
  - `text`: Target text.
  - `speaker`: Selection from preset voices (Aiden, Eric, Serena, etc.).
  - `instruct`: Optional style instructions.


## Installation

Ensure you have the required dependencies:
```bash
pip install torch torchaudio transformers librosa accelerate
```

## Tips for Best Results
- **Cloning**: Use clean, noise-free reference audio (5-15 seconds).
- **VRAM**: Use `bf16` precision to save significant memory with minimal quality loss.
- **Local Models**: Pre-download weights to `models/qwen-tts/` to prioritize local loading and avoid HuggingFace timeouts.

## 🔧 Integration Guide: Full Hybrid Morphographic Pipeline

When integrating Qwen3-TTS into a **real-time voice application** (VTuber, voice assistant, live streaming), use the full 3-stage hybrid morphographic pipeline for seamless, gap-free speech:

### Stage 1: Crossbar Compilation (`use_crossbar=True`)
Pass `use_crossbar=True` to `generate_voice_clone()` for optimized inference:
```python
with torch.inference_mode():
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        voice_clone_prompt=cached_prompt,
        non_streaming_mode=True,
        max_new_tokens=max(24, min(1024, len(text.split()) * 8)),
        temperature=0.6,
        use_crossbar=True,  # ← Hybrid Morphographic Compiler
    )
```

### Stage 2: Butterworth Artifact Smoothing (Per-Chunk)
Apply a 2nd-order Butterworth low-pass filter at 11kHz to remove crossbar artifacts:
```python
from scipy.signal import butter, sosfilt

_BUTTER_SOS = butter(2, 11000, btype='low', fs=24000, output='sos')

def butterworth_smooth(wav):
    """Apply 2nd-order Butterworth LPF at 11kHz + normalize to -3dB."""
    wav = sosfilt(_BUTTER_SOS, wav).astype(np.float32)
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav * (0.707 / peak)
    return wav

# Apply after generation:
audio = butterworth_smooth(wavs[0].astype(np.float32))
```

### Stage 3: Inter-Chunk Cosine Crossfade Blending
When streaming multi-sentence responses as separate chunks, blend chunk boundaries with a 50ms cosine crossfade to eliminate clicks and gaps:
```python
CROSSFADE_SAMPLES = int(24000 * 0.05)  # 1200 samples (50ms at 24kHz)
prev_chunk_tail = None

for i, chunk_audio in enumerate(generated_chunks):
    # Blend previous tail with current head
    if prev_chunk_tail is not None and len(chunk_audio) > CROSSFADE_SAMPLES * 2:
        head = chunk_audio[:CROSSFADE_SAMPLES]
        fade_in = (1 - np.cos(np.linspace(0, np.pi, CROSSFADE_SAMPLES))) / 2
        fade_out = 1 - fade_in
        blended = (prev_chunk_tail * fade_out + head * fade_in).astype(np.float32)
        chunk_audio = np.concatenate([blended, chunk_audio[CROSSFADE_SAMPLES:]])

    # Save tail for next crossfade
    if len(chunk_audio) > CROSSFADE_SAMPLES * 2:
        prev_chunk_tail = chunk_audio[-CROSSFADE_SAMPLES:].copy()
        # Play everything except tail (tail blends into next chunk)
        is_last = (i == len(generated_chunks) - 1)
        play(chunk_audio if is_last else chunk_audio[:-CROSSFADE_SAMPLES])
    else:
        prev_chunk_tail = None
        play(chunk_audio)
```

### Performance Tips
- **Warmup**: Run 5 short generations at startup to trigger CUDA graph capture and prime GPU clocks
- **Cache `voice_clone_prompt`**: Call `create_voice_clone_prompt()` once, reuse for all generations
- **BF16 precision**: Use `torch_dtype=torch.bfloat16` for best speed on RTX 30/40/50 series
- **Flash Attention 2**: Use `attn_implementation="flash_attention_2"` for 2x faster first audio
- **Token cap**: `max_new_tokens = max(24, min(1024, word_count * 8))` prevents wasted generation

### Result
Full pipeline achieves **RTF ~0.82** with zero quality loss and seamless chunk transitions — suitable for real-time conversational AI and VTuber applications.

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Official open-source repository by Alibaba Qwen team.

## License

- This project is licensed under the **Apache License 2.0**.
- Model weights are subject to the [Qwen3-TTS License Agreement](https://github.com/QwenLM/Qwen3-TTS#License).
