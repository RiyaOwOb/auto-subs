//! TEA-ASR and Qwen3-ForcedAligner transcription backend.
//!
//! Connects AutoSubs to taiwan-subtitle-windows AI models for Taiwanese Mandarin
//! transcription and precise word alignment.

use crate::types::{
    LabeledProgressFn, NewSegmentFn, ProgressType, Segment, SpeechSegment, TranscribeOptions,
    WordTimestamp,
};
use crate::utils::push_segment_clamped;
use eyre::{bail, eyre, Result};
use serde::Deserialize;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug, Deserialize)]
struct AdapterOutput {
    segments: Vec<RawSegment>,
    language: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawSegment {
    start: f64,
    end: f64,
    text: String,
    words: Option<Vec<WordTimestamp>>,
    speaker_id: Option<String>,
}

fn find_python() -> PathBuf {
    // 1. AUTOSUBS_PYTHON environment variable
    if let Ok(p) = std::env::var("AUTOSUBS_PYTHON") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return path;
        }
    }

    // 2. Well-known Windows paths
    #[cfg(target_os = "windows")]
    {
        for candidate in &[
            r"C:\Python314\python.exe",
            r"C:\Python312\python.exe",
            r"C:\Python311\python.exe",
            r"C:\Python310\python.exe",
        ] {
            let p = PathBuf::from(candidate);
            if p.is_file() {
                return p;
            }
        }
        if let Ok(local_appdata) = std::env::var("LOCALAPPDATA") {
            for ver in &["Python314", "Python313", "Python312", "Python311", "Python310"] {
                let p = PathBuf::from(&local_appdata)
                    .join("Programs")
                    .join("Python")
                    .join(ver)
                    .join("python.exe");
                if p.is_file() {
                    return p;
                }
            }
        }
    }

    // 3. System PATH check
    let which_cmd = if cfg!(windows) { "where" } else { "which" };
    if let Ok(output) = Command::new(which_cmd).arg("python").output() {
        if output.status.success() {
            if let Ok(text) = String::from_utf8(output.stdout) {
                if let Some(first) = text.lines().next() {
                    let p = PathBuf::from(first.trim());
                    if p.is_file() {
                        return p;
                    }
                }
            }
        }
    }

    PathBuf::from("python")
}

fn find_adapter_script() -> Result<PathBuf> {
    // 1. Relative to current executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p1 = dir.join("resources").join("taiwan_subtitle").join("taiwan_adapter.py");
            if p1.is_file() {
                return Ok(p1);
            }
            let p2 = dir.join("taiwan_subtitle").join("taiwan_adapter.py");
            if p2.is_file() {
                return Ok(p2);
            }
            if let Some(parent) = dir.parent() {
                let p3 = parent.join("resources").join("taiwan_subtitle").join("taiwan_adapter.py");
                if p3.is_file() {
                    return Ok(p3);
                }
            }
        }
    }

    // 2. Relative to current working directory
    for rel in &[
        "resources/taiwan_subtitle/taiwan_adapter.py",
        "src-tauri/resources/taiwan_subtitle/taiwan_adapter.py",
        "AutoSubs-App/src-tauri/resources/taiwan_subtitle/taiwan_adapter.py",
    ] {
        let p = PathBuf::from(rel);
        if p.is_file() {
            return Ok(p.canonicalize().unwrap_or(p));
        }
    }

    // 3. Known project scratch path
    let scratch_p = PathBuf::from(
        r"C:\Users\cel86\.gemini\antigravity\scratch\auto-subs\AutoSubs-App\src-tauri\resources\taiwan_subtitle\taiwan_adapter.py",
    );
    if scratch_p.is_file() {
        return Ok(scratch_p);
    }

    bail!("Could not find taiwan_adapter.py script in resources or search paths");
}

pub async fn transcribe_tea_asr(
    model_path: &Path,
    speech_segments: Vec<SpeechSegment>,
    options: &TranscribeOptions,
    _native_target: Option<&str>,
    use_gpu: Option<bool>,
    progress_callback: Option<&LabeledProgressFn>,
    new_segment_callback: Option<&NewSegmentFn>,
    abort_callback: Option<Box<dyn Fn() -> bool + Send + Sync>>,
) -> Result<(Vec<Segment>, Option<String>)> {
    tracing::info!("TEA-ASR: Starting transcription (model={})", options.model);

    if abort_callback.as_ref().map(|c| c()).unwrap_or(false) {
        bail!("Transcription cancelled");
    }

    if speech_segments.is_empty() {
        return Ok((Vec::new(), Some("zh".to_string())));
    }

    if let Some(cb) = progress_callback {
        cb(0, ProgressType::Analyze, "progressSteps.analyze.loading");
    }

    // Find Python and adapter script
    let python_bin = find_python();
    let adapter_script = find_adapter_script()?;

    tracing::info!(
        "TEA-ASR: Using Python at {:?}, adapter at {:?}",
        python_bin,
        adapter_script
    );

    // Reconstruct full audio buffer from speech segments
    let max_end = speech_segments
        .iter()
        .map(|s| s.end)
        .fold(0.0f64, f64::max);
    let total_samples = (max_end * 16000.0).ceil() as usize;
    let mut full_samples = vec![0i16; total_samples.max(16000)];

    for seg in &speech_segments {
        let start_idx = (seg.start * 16000.0).floor() as usize;
        for (i, &sample) in seg.samples.iter().enumerate() {
            if start_idx + i < full_samples.len() {
                full_samples[start_idx + i] = sample;
            }
        }
    }

    // Generate unique temp file paths
    let now_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();
    let temp_dir = std::env::temp_dir();
    let temp_wav = temp_dir.join(format!("autosubs_tea_asr_{}_{}.wav", pid, now_nanos));
    let temp_json = temp_dir.join(format!("autosubs_tea_asr_{}_{}.json", pid, now_nanos));

    crate::audio::write_wav(temp_wav.to_str().unwrap(), &full_samples)?;

    let user_offset = options.offset.unwrap_or(0.0);
    let model_arg = if model_path.is_file() || model_path.is_dir() {
        model_path.to_string_lossy().to_string()
    } else {
        options.model.clone()
    };

    let skip_align = options.enable_forced_alignment == Some(false);
    let backend_arg = match use_gpu {
        Some(true) => "cuda",
        Some(false) => "cpu",
        None => "auto",
    };

    let mut cmd = Command::new(&python_bin);
    cmd.arg(&adapter_script)
        .arg("--audio")
        .arg(&temp_wav)
        .arg("--model")
        .arg(&model_arg)
        .arg("--output-json")
        .arg(&temp_json)
        .arg("--backend")
        .arg(backend_arg);

    if skip_align {
        cmd.arg("--skip-align");
    }

    if let Some(ref adv) = options.advanced {
        if let Some(ref prompt) = adv.init_prompt {
            if !prompt.trim().is_empty() {
                for hw in prompt.split(',') {
                    let trimmed = hw.trim();
                    if !trimmed.is_empty() {
                        cmd.arg("--hotword").arg(trimmed);
                    }
                }
            }
        }
    }

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    tracing::info!("TEA-ASR: Spawning subprocess {:?}", cmd);
    let mut child = cmd.spawn().map_err(|e| eyre!("Failed to spawn Python process: {}", e))?;

    let stdout = child.stdout.take().ok_or_else(|| eyre!("Failed to capture stdout"))?;
    let stderr = child.stderr.take().ok_or_else(|| eyre!("Failed to capture stderr"))?;

    // Read stderr in background thread to prevent buffer deadlock
    let stderr_handle = std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        let mut err_lines = Vec::new();
        for line in reader.lines().flatten() {
            tracing::warn!("TEA-ASR py: {}", line);
            err_lines.push(line);
        }
        err_lines.join("\n")
    });

    let reader = BufReader::new(stdout);
    for line in reader.lines().flatten() {
        if abort_callback.as_ref().map(|c| c()).unwrap_or(false) {
            let _ = child.kill();
            let _ = std::fs::remove_file(&temp_wav);
            let _ = std::fs::remove_file(&temp_json);
            bail!("Transcription cancelled");
        }

        tracing::debug!("TEA-ASR stdout: {}", line);
        if line.starts_with("PROGRESS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let pct_str = parts[1].trim_end_matches('%');
                if let Ok(pct) = pct_str.parse::<i32>() {
                    let phase = if pct < 30 {
                        ProgressType::Analyze
                    } else if pct < 90 {
                        ProgressType::Transcribe
                    } else {
                        ProgressType::Finish
                    };
                    if let Some(cb) = progress_callback {
                        cb(pct, phase, "progressSteps.transcribe");
                    }
                }
            }
        }
    }

    let status = child.wait().map_err(|e| eyre!("Failed to wait on child process: {}", e))?;
    let stderr_output = stderr_handle.join().unwrap_or_default();

    let _ = std::fs::remove_file(&temp_wav);

    if !status.success() {
        let _ = std::fs::remove_file(&temp_json);
        bail!(
            "TEA-ASR process failed with exit code {:?}: {}",
            status.code(),
            stderr_output
        );
    }

    if !temp_json.exists() {
        bail!("TEA-ASR output JSON not generated: {}", temp_json.display());
    }

    let json_bytes = std::fs::read(&temp_json)?;
    let _ = std::fs::remove_file(&temp_json);

    let output: AdapterOutput = serde_json::from_slice(&json_bytes)
        .map_err(|e| eyre!("Failed to parse TEA-ASR JSON output: {}", e))?;

    let mut segments = Vec::new();
    for (i, raw) in output.segments.into_iter().enumerate() {
        let seg_start = raw.start + user_offset;
        let seg_end = raw.end + user_offset;

        let words = raw.words.map(|w_list| {
            w_list
                .into_iter()
                .map(|mut w| {
                    w.start += user_offset;
                    w.end += user_offset;
                    w
                })
                .collect()
        });

        let speaker_id = raw.speaker_id.or_else(|| {
            speech_segments
                .iter()
                .find(|s| s.start <= raw.start && raw.end <= s.end + 0.5)
                .and_then(|s| s.speaker_id.clone())
        });

        let segment = Segment {
            start: seg_start,
            end: seg_end,
            text: raw.text,
            words,
            speaker_id,
        };

        if let Some(cb) = new_segment_callback {
            cb(i, &segment, crate::types::SegmentStage::Transcribe);
        }

        push_segment_clamped(&mut segments, segment);
    }

    if let Some(cb) = progress_callback {
        cb(100, ProgressType::Finish, "progressSteps.finish");
    }

    let detected_lang = output.language.or_else(|| Some("zh".to_string()));
    Ok((segments, detected_lang))
}
