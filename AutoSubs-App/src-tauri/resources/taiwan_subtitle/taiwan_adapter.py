#!/usr/bin/env python3
"""
taiwan_adapter.py - Adapter connecting AutoSubs to taiwan-subtitle-windows AI models
Supports TEA-ASR (JacobLinCool/TEA-ASR-1.1-mini, JacobLinCool/TEA-ASR-1.1) and Qwen3-ForcedAligner
"""
import argparse
import json
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    import transcribe
except ImportError as e:
    print(f"Error importing transcribe module: {e}", file=sys.stderr)
    sys.exit(1)

MODEL_MAP = {
    "tea-asr-mini": "JacobLinCool/TEA-ASR-1.1-mini",
    "tea-asr": "JacobLinCool/TEA-ASR-1.1",
    "tea-asr-mlx-4bit": "Alkd/TEA-ASR-1.1-MLX-4bit",
    "qwen3-asr-1.7b": "mlx-community/Qwen3-ASR-1.7B-8bit",
    "qwen3-asr-0.6b": "mlx-community/Qwen3-ASR-0.6B-8bit",
    "qwen3-forced-aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
}

def resolve_model_id(model_name_or_path: str, default: str) -> str:
    if not model_name_or_path:
        return default
    mapped = MODEL_MAP.get(model_name_or_path.lower(), model_name_or_path)
    return mapped

def main():
    parser = argparse.ArgumentParser(description="AutoSubs Taiwan Subtitle Adapter")
    parser.add_argument("--audio", required=True, help="Path to input audio/video file")
    parser.add_argument("--model", default="JacobLinCool/TEA-ASR-1.1-mini", help="ASR model name, ID or path")
    parser.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B", help="Aligner model name, ID or path")
    parser.add_argument("--output-json", required=True, help="Path for output JSON file")
    parser.add_argument("--language", default="Chinese", help="Transcription language")
    parser.add_argument("--no-punctuation", action="store_true", help="Remove punctuation from subtitles")
    parser.add_argument("--skip-align", action="store_true", help="Skip forced alignment")
    parser.add_argument("--backend", default="auto", help="Backend (cuda, mlx, auto)")
    parser.add_argument("--hotword", action="append", default=[], help="Hotwords to boost")

    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    asr_model = resolve_model_id(args.model, "JacobLinCool/TEA-ASR-1.1-mini")
    aligner_model = resolve_model_id(args.aligner, "Qwen/Qwen3-ForcedAligner-0.6B") if not args.skip_align else None

    print("PROGRESS: 10% [Analyze] Starting Taiwan Subtitle pipeline...", flush=True)
    print(f"PROGRESS: 20% [Analyze] Model: {asr_model}", flush=True)

    temp_out_dir = Path(args.output_json).resolve().parent
    temp_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        hotwords = args.hotword if args.hotword else transcribe.HOTWORDS
        (srt_path, txt_path, json_path), bundle = transcribe.run_transcription(
            input_path=audio_path,
            output_dir=temp_out_dir,
            asr_model=asr_model,
            aligner_model=aligner_model,
            hotwords=hotwords,
            language=args.language,
            no_punctuation=args.no_punctuation,
            verbose=False,
            backend=args.backend,
        )

        print("PROGRESS: 90% [Finish] Converting transcript bundle...", flush=True)

        autosubs_segments = []
        all_words = bundle.words or []
        for seg in bundle.segments:
            s_start = float(seg.get("start", 0.0))
            s_end = float(seg.get("end", 0.0))
            s_text = seg.get("text", "")
            seg_words = [
                {
                    "text": w.get("text", ""),
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                }
                for w in all_words
                if s_start - 0.1 <= float(w.get("start", 0.0)) <= s_end + 0.1
            ]
            autosubs_segments.append({
                "start": s_start,
                "end": s_end,
                "text": s_text,
                "words": seg_words,
                "speaker_id": None
            })

        output_payload = {
            "segments": autosubs_segments,
            "language": "zh",
            "full_transcript": bundle.full_transcript,
            "processing_time": bundle.processing_time
        }

        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)

        print("PROGRESS: 100% [Finish] Transcription completed successfully.", flush=True)
        sys.exit(0)

    except Exception as exc:
        import traceback
        print(f"Error during transcription: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
