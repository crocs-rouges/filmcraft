//! Background transcription worker for FilmCraft using faster-whisper.
//!
//! Spawns the python bridge script in the background without blocking the UI thread,
//! displays real-time progress in the Text panel, and injects the resulting transcript
//! directly into the active session upon completion.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::Instant;

use filmcraft_project::{ItemId, ItemKind, MediaRef, Transcript};
use serde_json::Value;

use crate::FilmcraftApp;

pub struct TranscriptionJob {
    pub rx: Receiver<Result<Vec<(ItemId, Transcript)>, String>>,
    pub cancel_flag: Arc<AtomicBool>,
    pub start_time: Instant,
    pub clip_names: Vec<String>,
}

/// Helper to get the python command executable.
fn find_python_cmd() -> std::process::Command {
    if let Ok(py) = std::env::var("PYTHON") {
        let p = PathBuf::from(py);
        if p.exists() {
            return std::process::Command::new(p);
        }
    }
    let cand = PathBuf::from(r"C:\Users\romai\AppData\Local\Microsoft\WindowsApps\python.exe");
    if cand.exists() {
        std::process::Command::new(cand)
    } else {
        std::process::Command::new("python")
    }
}

/// Locates the `ai-bridge/filmcraft_transcript.py` script.
pub fn find_transcript_script() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("FILMCRAFT_TRANSCRIPT_SCRIPT") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let cand1 = parent.join("../../ai-bridge/filmcraft_transcript.py");
            if cand1.exists() {
                return Some(cand1.canonicalize().unwrap_or(cand1));
            }
            let cand2 = parent.join("ai-bridge/filmcraft_transcript.py");
            if cand2.exists() {
                return Some(cand2.canonicalize().unwrap_or(cand2));
            }
        }
    }
    let cand3 = PathBuf::from("ai-bridge/filmcraft_transcript.py");
    if cand3.exists() {
        return Some(cand3.canonicalize().unwrap_or(cand3));
    }
    let cand4 = PathBuf::from(r"C:\Users\romai\Documents\FilmCraft\ai-bridge\filmcraft_transcript.py");
    if cand4.exists() {
        return Some(cand4);
    }
    None
}

/// Whether a background transcription is currently in progress.
pub fn is_transcribing(app: &FilmcraftApp) -> bool {
    app.transcription_job.is_some()
}

/// Cancels any in-flight background transcription.
pub fn cancel_transcription(app: &mut FilmcraftApp) {
    if let Some(job) = app.transcription_job.take() {
        job.cancel_flag.store(true, Ordering::Relaxed);
        app.ui.status = "Transcription annulée.".into();
    }
}

/// Starts transcription of the target items in the background.
pub fn start_transcription(app: &mut FilmcraftApp, params: &Value) -> Result<(), String> {
    if app.transcription_job.is_some() {
        return Err("Une transcription est déjà en cours.".into());
    }

    let mut raw_items: Vec<ItemId> = Vec::new();
    if let Some(items_val) = params.get("items").and_then(Value::as_array) {
        for v in items_val {
            if let Some(id) = v.as_u64() {
                raw_items.push(ItemId(id));
            }
        }
    } else if let Some(item_val) = params.get("item").and_then(Value::as_u64) {
        raw_items.push(ItemId(item_val));
    }

    // If not explicitly passed, prefer active sequence audio clips
    if raw_items.is_empty() {
        if let Some(seq) = app.session.active_sequence() {
            for track in &seq.audio_tracks {
                for clip in &track.items {
                    if clip.enabled {
                        raw_items.push(clip.item);
                    }
                }
            }
        }
    }

    // Fallback to project selection
    if raw_items.is_empty() {
        raw_items = app.session.state.project_selection.clone();
    }

    let mut target_items: Vec<ItemId> = Vec::new();
    for i in raw_items {
        if let Some((root, _, _)) = app.session.project.resolve_media(i) {
            if !target_items.contains(&root) {
                target_items.push(root);
            }
        }
    }

    let mut jobs: Vec<(ItemId, String, PathBuf)> = Vec::new();
    for id in target_items {
        if let Some(it) = app.session.project.item(id) {
            if let ItemKind::Media(m) = &it.kind {
                if let MediaRef::File { path } = &m.media {
                    let mut p = PathBuf::from(path);
                    if !p.exists() {
                        if let Some(proj_path) = &app.session.path {
                            if let Some(dir) = PathBuf::from(proj_path).parent() {
                                let cand = dir.join(&p);
                                if cand.exists() {
                                    p = cand;
                                }
                            }
                        }
                    }
                    if p.exists() {
                        jobs.push((id, it.name.clone(), p));
                    } else {
                        log::warn!("Fichier média introuvable sur le disque: {:?}", p);
                    }
                }
            }
        }
    }

    if jobs.is_empty() {
        return Err("Aucun fichier média source trouvé sur le disque pour ces clips.".into());
    }

    let script_path = find_transcript_script().ok_or_else(|| {
        "Script ai-bridge/filmcraft_transcript.py introuvable.".to_string()
    })?;

    let ma = &app.session.prefs.media_analysis;
    let raw_model = params.get("model").and_then(Value::as_str).unwrap_or(&ma.whisper_model);
    let model = raw_model.strip_prefix("whisper-").unwrap_or(raw_model).to_string();
    let language = params.get("language").and_then(Value::as_str).map(String::from).unwrap_or_else(|| {
        if ma.language_auto_detect || ma.default_language == "en" {
            "fr".into()
        } else {
            ma.default_language.clone()
        }
    });
    let force_whisper = params.get("force").and_then(Value::as_bool).unwrap_or(true);

    let (tx, rx) = mpsc::channel();
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let cancel_thread = cancel_flag.clone();
    let clip_names: Vec<String> = jobs.iter().map(|(_, name, _)| name.clone()).collect();

    log::info!("Démarrage de la transcription d'arrière-plan pour {} média(s)", jobs.len());

    std::thread::spawn(move || {
        let mut done: Vec<(ItemId, Transcript)> = Vec::new();
        for (item_id, _name, media_path) in jobs {
            if cancel_thread.load(Ordering::Relaxed) {
                let _ = tx.send(Err("Transcription annulée.".into()));
                return;
            }

            let temp_json = std::env::temp_dir().join(format!(
                "filmcraft_tr_{}_{}.json",
                item_id.0,
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
            ));

            let mut cmd = find_python_cmd();
            #[cfg(target_os = "windows")]
            {
                use std::os::windows::process::CommandExt;
                const CREATE_NO_WINDOW: u32 = 0x08000000;
                cmd.creation_flags(CREATE_NO_WINDOW);
            }

            cmd.arg(&script_path)
                .arg("--media")
                .arg(&media_path)
                .arg("--item-id")
                .arg(item_id.0.to_string())
                .arg("--export-json")
                .arg(&temp_json)
                .arg("--model")
                .arg(&model)
                .arg("--language")
                .arg(&language);

            if force_whisper {
                cmd.arg("--force-whisper");
            }

            log::info!("Exécution de faster-whisper sur {:?}", media_path);

            let output = match cmd.output() {
                Ok(out) => out,
                Err(e) => {
                    let err_msg = format!("Impossible d'exécuter python: {e}");
                    log::error!("{err_msg}");
                    let _ = tx.send(Err(err_msg));
                    return;
                }
            };

            if cancel_thread.load(Ordering::Relaxed) {
                let _ = std::fs::remove_file(&temp_json);
                let _ = tx.send(Err("Transcription annulée.".into()));
                return;
            }

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                let err_msg = if !stderr.trim().is_empty() { stderr.to_string() } else { stdout.to_string() };
                log::error!("Erreur script transcription: {}", err_msg);
                let _ = std::fs::remove_file(&temp_json);
                let _ = tx.send(Err(format!("Erreur transcription: {err_msg}")));
                return;
            }

            let json_bytes = match std::fs::read(&temp_json) {
                Ok(bytes) => bytes,
                Err(e) => {
                    let err_msg = format!("Fichier de transcription introuvable: {e}");
                    log::error!("{err_msg}");
                    let _ = tx.send(Err(err_msg));
                    return;
                }
            };
            let _ = std::fs::remove_file(&temp_json);

            let data: Value = match serde_json::from_slice(&json_bytes) {
                Ok(val) => val,
                Err(e) => {
                    let err_msg = format!("Format JSON invalide: {e}");
                    log::error!("{err_msg}");
                    let _ = tx.send(Err(err_msg));
                    return;
                }
            };

            let tr_val = data.get("transcript").cloned().unwrap_or(data);
            let mut transcript: Transcript = match serde_json::from_value(tr_val) {
                Ok(t) => t,
                Err(e) => {
                    let err_msg = format!("Données de transcription invalides: {e}");
                    log::error!("{err_msg}");
                    let _ = tx.send(Err(err_msg));
                    return;
                }
            };
            transcript.normalize();
            done.push((item_id, transcript));
        }

        let _ = tx.send(Ok(done));
    });

    app.transcription_job = Some(TranscriptionJob {
        rx,
        cancel_flag,
        start_time: Instant::now(),
        clip_names,
    });
    app.ui.status = "Transcription en arrière-plan démarrée avec faster-whisper...".into();
    Ok(())
}

/// Polls in-flight background transcription and inserts completed transcripts into the project.
pub fn poll_transcription(app: &mut FilmcraftApp, ctx: &egui::Context) {
    let Some(job) = &mut app.transcription_job else { return };
    match job.rx.try_recv() {
        Ok(Ok(items)) => {
            let total_words: usize = items.iter().map(|(_, t)| t.words.len()).sum();
            let r = app.session.edit("Transcribe", move |pr, _| {
                for (item, tr) in items {
                    pr.transcripts.insert(item, Arc::new(tr));
                }
                Ok(())
            });
            if let Err(e) = r {
                app.ui.status = format!("Erreur lors de l'enregistrement de la transcription: {e}");
            } else {
                app.ui.status = format!("Transcription terminée avec succès ({} mots) !", total_words);
            }
            app.transcription_job = None;
            ctx.request_repaint();
        }
        Ok(Err(err)) => {
            app.ui.status = err;
            app.transcription_job = None;
            ctx.request_repaint();
        }
        Err(TryRecvError::Empty) => {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
        Err(TryRecvError::Disconnected) => {
            app.ui.status = "Le thread de transcription s'est arrêté inopinément.".into();
            app.transcription_job = None;
            ctx.request_repaint();
        }
    }
}
