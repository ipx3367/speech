#!/usr/bin/env python3
from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sounddevice as sd
import torch
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from transformers import pipeline

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_MS = 200
BLOCK_SIZE = SAMPLE_RATE * BLOCK_MS // 1000
DRAFT_MODELS = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
]
BREEZE_MODELS = [
    "MediaTek-Research/Breeze-ASR-26",
]


@dataclass
class Segment:
    segment_id: int
    audio: np.ndarray
    created_at: float = field(default_factory=time.time)


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current_draft: str = ""
        self.final_lines: list[str] = []
        self.pending_ids: set[int] = set()
        self.running = False
        self.status_text = "尚未啟動"
        self.last_error = ""

    def set_draft(self, text: str) -> None:
        with self.lock:
            self.current_draft = text.strip()

    def add_pending(self, segment_id: int) -> None:
        with self.lock:
            self.pending_ids.add(segment_id)

    def finalize(self, segment_id: int, text: str) -> None:
        with self.lock:
            self.pending_ids.discard(segment_id)
            clean = text.strip()
            if clean:
                self.final_lines.append(clean)
            self.current_draft = ""

    def set_status(self, text: str) -> None:
        with self.lock:
            self.status_text = text

    def set_error(self, text: str) -> None:
        with self.lock:
            self.last_error = text
            self.status_text = f"錯誤：{text}"

    def clear(self) -> None:
        with self.lock:
            self.current_draft = ""
            self.final_lines = []
            self.pending_ids = set()
            self.last_error = ""
            self.status_text = "尚未啟動"

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "draft": self.current_draft,
                "final_lines": list(self.final_lines),
                "pending": sorted(self.pending_ids),
                "running": self.running,
                "status_text": self.status_text,
                "last_error": self.last_error,
            }


def pick_device_for_torch() -> tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_vad_model():
    model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def load_draft_asr(model_name: str):
    device, dtype = pick_device_for_torch()
    return pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        chunk_length_s=10,
        torch_dtype=dtype,
        device=device,
    )


def load_breeze_asr(model_name: str):
    device, dtype = pick_device_for_torch()
    return pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        chunk_length_s=20,
        torch_dtype=dtype,
        device=device,
    )


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / 32768.0
    return np.clip(audio, -1.0, 1.0)


def transcribe_audio(pipe, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    audio = normalize_audio(audio)
    result = pipe({"array": audio, "sampling_rate": sample_rate})
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()


class AudioCoordinator:
    def __init__(
        self,
        state: SharedState,
        draft_pipe,
        breeze_pipe,
        vad_model,
        get_speech_timestamps,
        min_speech_ms: int,
        max_segment_s: float,
        silence_end_ms: int,
        draft_update_s: float,
        input_device: Optional[int],
    ) -> None:
        self.state = state
        self.draft_pipe = draft_pipe
        self.breeze_pipe = breeze_pipe
        self.vad_model = vad_model
        self.get_speech_timestamps = get_speech_timestamps
        self.min_speech_ms = min_speech_ms
        self.max_segment_samples = int(max_segment_s * SAMPLE_RATE)
        self.silence_end_blocks = max(1, silence_end_ms // BLOCK_MS)
        self.draft_update_blocks = max(1, int(draft_update_s * 1000 // BLOCK_MS))
        self.input_device = input_device

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self.refine_queue: queue.Queue[Segment] = queue.Queue()
        self.segment_id = 0

        self.current_segment: list[np.ndarray] = []
        self.silence_blocks = 0
        self.blocks_since_draft = 0
        self.in_speech = False
        self.stream: sd.InputStream | None = None
        self.stop_event = threading.Event()
        self.work_threads: list[threading.Thread] = []

    def audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.state.set_status(f"音訊狀態：{status}")
        chunk = np.copy(indata[:, 0]).astype(np.float32)
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            pass

    def start(self) -> None:
        self.state.running = True
        self.state.set_status("載入音訊串流…")
        self.stop_event.clear()
        self.work_threads = [
            threading.Thread(target=self.processing_worker, daemon=True),
            threading.Thread(target=self.refine_worker, daemon=True),
        ]
        for t in self.work_threads:
            t.start()

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            device=self.input_device,
            callback=self.audio_callback,
        )
        self.stream.start()
        self.state.set_status("錄音中… 請開始說話")

    def stop(self) -> None:
        self.stop_event.set()
        self.state.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.flush_segment(force=True)
        self.state.set_status("已停止")

    def processing_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            self.process_chunk(chunk)

    def process_chunk(self, chunk: np.ndarray) -> None:
        is_speech = self.chunk_has_speech(chunk)

        if is_speech:
            self.in_speech = True
            self.silence_blocks = 0
            self.current_segment.append(chunk)
            self.blocks_since_draft += 1

            if self.blocks_since_draft >= self.draft_update_blocks:
                self.blocks_since_draft = 0
                self.update_draft()

            if self.current_segment_samples >= self.max_segment_samples:
                self.flush_segment(force=True)
        else:
            if self.in_speech:
                self.current_segment.append(chunk)
                self.silence_blocks += 1
                if self.silence_blocks >= self.silence_end_blocks:
                    self.flush_segment(force=False)

    @property
    def current_segment_samples(self) -> int:
        return sum(len(c) for c in self.current_segment)

    def chunk_has_speech(self, chunk: np.ndarray) -> bool:
        audio = torch.from_numpy(normalize_audio(chunk))
        speech = self.get_speech_timestamps(
            audio,
            self.vad_model,
            sampling_rate=SAMPLE_RATE,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=50,
            threshold=0.35,
        )
        return len(speech) > 0

    def update_draft(self) -> None:
        if not self.current_segment:
            return
        audio = np.concatenate(self.current_segment, axis=0)
        if len(audio) < SAMPLE_RATE // 2:
            return
        try:
            text = transcribe_audio(self.draft_pipe, audio)
        except Exception as exc:  # noqa: BLE001
            text = f"[draft error] {exc}"
        self.state.set_draft(text)

    def flush_segment(self, force: bool) -> None:
        if not self.current_segment:
            self.reset_segment_state()
            return

        audio = np.concatenate(self.current_segment, axis=0)
        self.reset_segment_state()

        min_samples = SAMPLE_RATE
        if len(audio) < min_samples and not force:
            return

        self.segment_id += 1
        seg = Segment(segment_id=self.segment_id, audio=audio)
        self.state.add_pending(seg.segment_id)
        self.refine_queue.put(seg)

    def reset_segment_state(self) -> None:
        self.current_segment = []
        self.silence_blocks = 0
        self.blocks_since_draft = 0
        self.in_speech = False
        self.state.set_draft("")

    def refine_worker(self) -> None:
        while not self.stop_event.is_set() or not self.refine_queue.empty():
            try:
                seg = self.refine_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self.state.set_status(f"Breeze 精修中… #{seg.segment_id}")
                text = transcribe_audio(self.breeze_pipe, seg.audio)
            except Exception as exc:  # noqa: BLE001
                text = f"[breeze error] {exc}"
            self.state.finalize(seg.segment_id, text)
            if not self.stop_event.is_set():
                self.state.set_status("錄音中… 請開始說話")


class App:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.state = SharedState()
        self.coordinator: AudioCoordinator | None = None
        self.backend_thread: threading.Thread | None = None
        self.device_map: dict[str, int] = {}

        self.root.title("Breeze 即時字幕 GUI")
        self.root.geometry("1000x720")

        self.status_var = tk.StringVar(value="準備中…")
        self.device_var = tk.StringVar()
        self.draft_model_var = tk.StringVar(value=args.draft_model)
        self.breeze_model_var = tk.StringVar(value=BREEZE_MODELS[0])
        self.min_speech_var = tk.StringVar(value=str(args.min_speech_ms))
        self.silence_end_var = tk.StringVar(value=str(args.silence_end_ms))
        self.draft_update_var = tk.StringVar(value=str(args.draft_update_s))
        self.max_segment_var = tk.StringVar(value=str(args.max_segment_s))

        self._build_ui()
        self.refresh_devices()
        self.poll_state()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill=tk.X)

        ttk.Label(top, text="麥克風").grid(row=0, column=0, sticky=tk.W)
        self.device_combo = ttk.Combobox(top, textvariable=self.device_var, state="readonly", width=48)
        self.device_combo.grid(row=0, column=1, sticky=tk.W, padx=6)
        ttk.Button(top, text="重新整理", command=self.refresh_devices).grid(row=0, column=2, padx=6)

        ttk.Label(top, text="草稿模型").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        self.draft_model_combo = ttk.Combobox(
            top,
            textvariable=self.draft_model_var,
            state="readonly",
            width=48,
            values=DRAFT_MODELS,
        )
        self.draft_model_combo.grid(row=1, column=1, sticky=tk.W, padx=6, pady=(8, 0))

        ttk.Label(top, text="精修模型").grid(row=2, column=0, sticky=tk.W, pady=(8, 0))
        self.breeze_model_combo = ttk.Combobox(
            top,
            textvariable=self.breeze_model_var,
            state="readonly",
            width=48,
            values=BREEZE_MODELS,
        )
        self.breeze_model_combo.grid(row=2, column=1, sticky=tk.W, padx=6, pady=(8, 0))

        ttk.Label(top, text="min speech ms").grid(row=3, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(top, textvariable=self.min_speech_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=6, pady=(8, 0))

        ttk.Label(top, text="silence end ms").grid(row=3, column=2, sticky=tk.W, pady=(8, 0))
        ttk.Entry(top, textvariable=self.silence_end_var, width=10).grid(row=3, column=3, sticky=tk.W, padx=6, pady=(8, 0))

        ttk.Label(top, text="draft update s").grid(row=4, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(top, textvariable=self.draft_update_var, width=10).grid(row=4, column=1, sticky=tk.W, padx=6, pady=(8, 0))

        ttk.Label(top, text="max segment s").grid(row=4, column=2, sticky=tk.W, pady=(8, 0))
        ttk.Entry(top, textvariable=self.max_segment_var, width=10).grid(row=4, column=3, sticky=tk.W, padx=6, pady=(8, 0))

        btns = ttk.Frame(self.root, padding=(12, 0, 12, 0))
        btns.pack(fill=tk.X)
        self.start_btn = ttk.Button(btns, text="開始", command=self.start_engine)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(btns, text="停止", command=self.stop_engine, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="清空文字", command=self.clear_text).pack(side=tk.LEFT)

        status_frame = ttk.Frame(self.root, padding=(12, 8, 12, 0))
        status_frame.pack(fill=tk.X)
        ttk.Label(status_frame, text="狀態：").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)

        draft_frame = ttk.LabelFrame(self.root, text="即時草稿", padding=10)
        draft_frame.pack(fill=tk.BOTH, expand=False, padx=12, pady=(10, 6))
        self.draft_text = ScrolledText(draft_frame, height=6, wrap=tk.WORD, font=("Arial Unicode MS", 16))
        self.draft_text.pack(fill=tk.BOTH, expand=True)
        self.draft_text.configure(state=tk.DISABLED)

        final_frame = ttk.LabelFrame(self.root, text="正式字幕（Breeze 精修）", padding=10)
        final_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6, 12))
        self.final_text = ScrolledText(final_frame, height=16, wrap=tk.WORD, font=("Arial Unicode MS", 18))
        self.final_text.pack(fill=tk.BOTH, expand=True)
        self.final_text.configure(state=tk.DISABLED)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def refresh_devices(self) -> None:
        self.device_map.clear()
        names: list[str] = []
        try:
            devices = sd.query_devices()
        except Exception as exc:  # noqa: BLE001
            self.status_var.set(f"讀取音訊裝置失敗：{exc}")
            return

        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                label = f"{idx}: {dev['name']}"
                names.append(label)
                self.device_map[label] = idx

        self.device_combo["values"] = names
        if names and not self.device_var.get():
            self.device_var.set(names[0])
        elif self.args.input_device is not None:
            for label, idx in self.device_map.items():
                if idx == self.args.input_device:
                    self.device_var.set(label)
                    break

    def clear_text(self) -> None:
        self.state.clear()
        self.status_var.set("已清空")
        self._set_text(self.draft_text, "")
        self._set_text(self.final_text, "")

    def start_engine(self) -> None:
        if self.coordinator is not None:
            return
        try:
            min_speech_ms = int(self.min_speech_var.get())
            silence_end_ms = int(self.silence_end_var.get())
            draft_update_s = float(self.draft_update_var.get())
            max_segment_s = float(self.max_segment_var.get())
        except ValueError:
            messagebox.showerror("參數錯誤", "請確認參數都是有效數字。")
            return

        label = self.device_var.get().strip()
        input_device = self.device_map.get(label)
        draft_model = self.draft_model_var.get().strip() or self.args.draft_model
        breeze_model = self.breeze_model_var.get().strip() or BREEZE_MODELS[0]

        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("載入模型中… 第一次會比較久")
        self.state.clear()

        def runner() -> None:
            try:
                self.state.set_status("載入 VAD…")
                vad_model, get_speech_timestamps = load_vad_model()
                self.state.set_status("載入即時草稿模型…")
                draft_pipe = load_draft_asr(draft_model)
                self.state.set_status("載入 Breeze-ASR-26…")
                breeze_pipe = load_breeze_asr(breeze_model)
                self.coordinator = AudioCoordinator(
                    state=self.state,
                    draft_pipe=draft_pipe,
                    breeze_pipe=breeze_pipe,
                    vad_model=vad_model,
                    get_speech_timestamps=get_speech_timestamps,
                    min_speech_ms=min_speech_ms,
                    max_segment_s=max_segment_s,
                    silence_end_ms=silence_end_ms,
                    draft_update_s=draft_update_s,
                    input_device=input_device,
                )
                self.coordinator.start()
            except Exception as exc:  # noqa: BLE001
                self.state.set_error(str(exc))
                self.coordinator = None
                self.root.after(0, self._restore_stopped_buttons)

        self.backend_thread = threading.Thread(target=runner, daemon=True)
        self.backend_thread.start()

    def stop_engine(self) -> None:
        if self.coordinator is not None:
            self.coordinator.stop()
            self.coordinator = None
        self._restore_stopped_buttons()

    def _restore_stopped_buttons(self) -> None:
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)

    def poll_state(self) -> None:
        snap = self.state.snapshot()
        self.status_var.set(snap["status_text"])
        self._set_text(self.draft_text, snap["draft"])
        self._set_text(self.final_text, "\n".join(snap["final_lines"]))
        if not snap["running"] and self.coordinator is None:
            self._restore_stopped_buttons()
        self.root.after(200, self.poll_state)

    def _set_text(self, widget: ScrolledText, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        current = widget.get("1.0", tk.END).strip()
        if current != text.strip():
            widget.delete("1.0", tk.END)
            if text:
                widget.insert(tk.END, text)
                widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def on_close(self) -> None:
        try:
            self.stop_engine()
        finally:
            self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Breeze 即時字幕 GUI")
    parser.add_argument("--input-device", type=int, default=None, help="sounddevice 輸入裝置 index")
    parser.add_argument("--draft-model", default="openai/whisper-small", help="即時草稿模型")
    parser.add_argument("--min-speech-ms", type=int, default=15, help="VAD 最小語音長度")
    parser.add_argument("--max-segment-s", type=float, default=14.0, help="單段最長秒數")
    parser.add_argument("--silence-end-ms", type=int, default=250, help="停頓多久視為一句結束")
    parser.add_argument("--draft-update-s", type=float, default=0.25, help="多久更新一次草稿")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    App(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())