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
# 可選的即時草稿模型。模型越小延遲通常越低，但辨識品質也可能下降。
DRAFT_MODELS = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
]
# 正式精修模型目前固定使用 Breeze-ASR-26。
BREEZE_MODELS = [
    "MediaTek-Research/Breeze-ASR-26",
]


@dataclass
class Segment:
    """送往 Breeze 精修的完整語音片段。"""

    segment_id: int
    audio: np.ndarray
    created_at: float = field(default_factory=time.time)


@dataclass
class DraftRequest:
    """送往草稿辨識 worker 的請求。

    revision 用來辨識這份草稿是否仍屬於目前的片段。
    如果片段已經 flush/reset，舊草稿結果回來時就會被丟棄，避免 UI 顯示過期內容。
    """

    revision: int
    audio: np.ndarray


class SharedState:
    """跨執行緒共享的 UI 狀態容器。

    Tkinter 元件只能在主執行緒安全更新，因此背景執行緒只寫入這個狀態物件，
    由 GUI 的 poll loop 定期讀取 snapshot，再同步到畫面上。
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current_draft: str = ""
        self.final_lines: list[str] = []
        self.pending_ids: set[int] = set()
        self.running = False
        self.status_text = "尚未啟動"
        self.last_error = ""
        self.audio_level = 0.0

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

    def set_audio_level(self, level: float) -> None:
        with self.lock:
            self.audio_level = max(0.0, min(100.0, float(level)))

    def clear(self) -> None:
        with self.lock:
            self.current_draft = ""
            self.final_lines = []
            self.pending_ids = set()
            self.last_error = ""
            self.status_text = "尚未啟動"
            self.audio_level = 0.0

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "draft": self.current_draft,
                "final_lines": list(self.final_lines),
                "pending": sorted(self.pending_ids),
                "running": self.running,
                "status_text": self.status_text,
                "last_error": self.last_error,
                "audio_level": self.audio_level,
            }


def pick_device_for_torch() -> tuple[str, torch.dtype]:
    """選擇推論裝置與對應 dtype。

    在 Apple Silicon 上優先使用 MPS，以降低延遲；
    否則退回 CPU 與 float32，避免不相容的 dtype 組合。
    """

    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_vad_model():
    """載入 Silero VAD 與其工具函式。"""

    model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def load_draft_asr(model_name: str):
    """建立即時草稿 ASR pipeline。"""

    device, dtype = pick_device_for_torch()
    return pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        chunk_length_s=10,
        torch_dtype=dtype,
        device=device,
    )


def load_breeze_asr(model_name: str):
    """建立 Breeze 正式字幕精修 pipeline。"""

    device, dtype = pick_device_for_torch()
    return pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        chunk_length_s=20,
        torch_dtype=dtype,
        device=device,
    )


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """將音訊整理成單聲道 float32 並限制在 [-1, 1]。

    有些來源可能是多聲道或 int16 範圍，這裡先做最小必要轉換，
    讓後續 VAD 與 ASR 都能吃到一致格式。
    """

    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / 32768.0
    return np.clip(audio, -1.0, 1.0)


def transcribe_audio(pipe, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """以 transformers pipeline 進行單段語音辨識。"""

    audio = normalize_audio(audio)
    result = pipe({"array": audio, "sampling_rate": sample_rate})
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()


class ModelCache:
    """模型快取。

    這個專案最大的冷啟動成本來自模型下載與載入。
    透過快取，同一個模型名稱在同一個 App 生命週期內只會初始化一次。
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.vad_bundle = None
        self.draft_pipes: dict[str, object] = {}
        self.breeze_pipes: dict[str, object] = {}

    def get_vad(self):
        with self.lock:
            if self.vad_bundle is None:
                self.vad_bundle = load_vad_model()
            return self.vad_bundle

    def get_draft_pipe(self, model_name: str):
        with self.lock:
            pipe = self.draft_pipes.get(model_name)
            if pipe is None:
                pipe = load_draft_asr(model_name)
                self.draft_pipes[model_name] = pipe
            return pipe

    def get_breeze_pipe(self, model_name: str):
        with self.lock:
            pipe = self.breeze_pipes.get(model_name)
            if pipe is None:
                pipe = load_breeze_asr(model_name)
                self.breeze_pipes[model_name] = pipe
            return pipe


class AudioCoordinator:
    """協調收音、VAD 分段、草稿辨識與 Breeze 精修。

    執行緒分工如下：
    1. sounddevice callback: 只負責把音訊塊塞進 queue，避免在 callback 做重工作。
    2. processing_worker: 做 VAD、片段累積、判斷何時送草稿/正式精修。
    3. draft_worker: 專門處理較頻繁的草稿辨識，避免阻塞 processing_worker。
    4. refine_worker: 處理單段完成後的 Breeze 精修。
    """

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

        # audio_queue 用來承接即時音訊塊；draft_queue 只保留最新草稿請求；
        # refine_queue 則保留完整片段，等 Breeze 逐段精修。
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self.draft_queue: queue.Queue[DraftRequest] = queue.Queue(maxsize=1)
        self.refine_queue: queue.Queue[Segment] = queue.Queue()
        self.segment_id = 0

        # current_segment 保存目前正在講的一句話音訊塊。
        # current_segment_samples_count 避免每次用 sum(len(...)) 重算整段長度。
        self.current_segment: list[np.ndarray] = []
        self.current_segment_samples_count = 0
        self.silence_blocks = 0
        self.blocks_since_draft = 0
        self.in_speech = False
        # segment_revision 會在片段重置時遞增，用來識別過期 draft 結果。
        self.segment_revision = 0
        self.state_lock = threading.Lock()
        self.stream: sd.InputStream | None = None
        self.stop_event = threading.Event()
        self.work_threads: list[threading.Thread] = []

    def audio_callback(self, indata, frames, time_info, status) -> None:
        """sounddevice 的即時回呼。

        這裡只做最輕量的複製與入列；若 queue 滿了就直接丟棄，避免 callback 卡住。
        真正的 VAD / ASR 一律在背景 worker 做。
        """

        if status:
            self.state.set_status(f"音訊狀態：{status}")
        chunk = np.copy(indata[:, 0]).astype(np.float32)
        self.state.set_audio_level(self.calculate_audio_level(chunk))
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            pass

    def start(self) -> None:
        """啟動背景 worker 與麥克風輸入串流。"""

        self.state.running = True
        self.state.set_status("載入音訊串流…")
        self.stop_event.clear()
        self.work_threads = [
            threading.Thread(target=self.processing_worker, daemon=True),
            threading.Thread(target=self.draft_worker, daemon=True),
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
        """停止錄音並嘗試把最後一段未送出的語音送去精修。"""

        self.stop_event.set()
        self.state.running = False
        self.state.set_audio_level(0.0)
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
        """持續消化 audio_queue，執行 VAD 與分段控制。"""

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            self.process_chunk(chunk)

    def process_chunk(self, chunk: np.ndarray) -> None:
        """處理單一音訊塊。

        如果判定為語音，就把它併入當前片段，並依設定頻率排入草稿辨識。
        如果遇到靜音，則累積靜音長度；靜音足夠久就視為一句結束並 flush。
        """

        is_speech = self.chunk_has_speech(chunk)

        if is_speech:
            self.in_speech = True
            self.silence_blocks = 0
            self.current_segment.append(chunk)
            self.current_segment_samples_count += len(chunk)
            self.blocks_since_draft += 1

            if self.blocks_since_draft >= self.draft_update_blocks:
                self.blocks_since_draft = 0
                self.enqueue_draft_update()

            if self.current_segment_samples >= self.max_segment_samples:
                self.flush_segment(force=True)
        else:
            if self.in_speech:
                self.current_segment.append(chunk)
                self.current_segment_samples_count += len(chunk)
                self.silence_blocks += 1
                if self.silence_blocks >= self.silence_end_blocks:
                    self.flush_segment(force=False)

    @property
    def current_segment_samples(self) -> int:
        return self.current_segment_samples_count

    def chunk_has_speech(self, chunk: np.ndarray) -> bool:
        """用 VAD 判定單一音訊塊是否含語音。"""

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

    def calculate_audio_level(self, chunk: np.ndarray) -> float:
        """將音訊塊振幅轉成 0-100 的 UI 音量值。

        原始麥克風輸入通常落在很小的浮點範圍，因此這裡同時參考 peak 與 RMS，
        並做適度放大，讓 GUI 上的動態更容易觀察。
        """

        normalized = normalize_audio(chunk)
        if normalized.size == 0:
            return 0.0
        peak = float(np.max(np.abs(normalized)))
        rms = float(np.sqrt(np.mean(np.square(normalized))))
        level = max(peak * 140.0, rms * 220.0)
        return min(100.0, level)

    def enqueue_draft_update(self) -> None:
        """送出最新草稿辨識請求。

        draft_queue 大小固定為 1。若草稿 worker 跟不上，舊請求會被更新的內容覆蓋，
        這樣可以確保使用者看到的是「最新一句話的最新版本」，而不是排隊中的過時草稿。
        """

        if not self.current_segment:
            return
        audio = np.concatenate(self.current_segment, axis=0)
        if len(audio) < SAMPLE_RATE // 2:
            return
        req = DraftRequest(revision=self.segment_revision, audio=audio)
        try:
            self.draft_queue.put_nowait(req)
        except queue.Full:
            try:
                self.draft_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.draft_queue.put_nowait(req)
            except queue.Full:
                pass

    def draft_worker(self) -> None:
        """專門處理草稿辨識。

        由於草稿更新頻率高，如果直接在 processing_worker 裡推論，
        會拖慢 VAD 與 queue 消化速度。拆成獨立 worker 可以讓收音流程更穩定。
        """

        while not self.stop_event.is_set():
            try:
                req = self.draft_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                text = transcribe_audio(self.draft_pipe, req.audio)
            except Exception as exc:  # noqa: BLE001
                text = f"[draft error] {exc}"
            with self.state_lock:
                # 只接受仍屬於當前片段版本的草稿，避免 reset 後舊結果覆寫新狀態。
                if req.revision == self.segment_revision and not self.stop_event.is_set():
                    self.state.set_draft(text)

    def flush_segment(self, force: bool) -> None:
        """將目前片段封裝成 Segment，送往 Breeze 精修。"""

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
        """清空目前片段與相關計數器。"""

        self.current_segment = []
        self.current_segment_samples_count = 0
        self.silence_blocks = 0
        self.blocks_since_draft = 0
        self.in_speech = False
        with self.state_lock:
            self.segment_revision += 1
        self.state.set_draft("")

    def refine_worker(self) -> None:
        """逐段執行 Breeze 精修。

        refine_queue 在 stop 後仍會繼續清空，確保已經送出的片段可以完成辨識。
        """

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
    """Tkinter GUI 主程式。"""

    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.state = SharedState()
        # ModelCache 與 App 同生命週期，讓多次開始/停止可以重用已載入模型。
        self.model_cache = ModelCache()
        self.coordinator: AudioCoordinator | None = None
        self.backend_thread: threading.Thread | None = None
        self.device_map: dict[str, int] = {}
        # rendered_* 用來記錄目前畫面上已經顯示的內容，便於做增量更新。
        self.rendered_draft = ""
        self.rendered_final_lines: list[str] = []

        self.root.title("Breeze 即時字幕 GUI")
        self.root.geometry("1000x720")

        self.status_var = tk.StringVar(value="準備中…")
        self.audio_level_var = tk.DoubleVar(value=0.0)
        self.device_var = tk.StringVar()
        self.draft_model_var = tk.StringVar(value=args.draft_model)
        self.breeze_model_var = tk.StringVar(value=BREEZE_MODELS[0])
        self.min_speech_var = tk.StringVar(value=str(args.min_speech_ms))
        self.silence_end_var = tk.StringVar(value=str(args.silence_end_ms))
        self.draft_update_var = tk.StringVar(value=str(args.draft_update_s))
        self.max_segment_var = tk.StringVar(value=str(args.max_segment_s))

        self._configure_styles()
        self._build_ui()
        self.refresh_devices()
        self.poll_state()

    def _configure_styles(self) -> None:
        """設定 GUI 樣式。"""

        style = ttk.Style(self.root)
        style.configure(
            "AudioLevel.Horizontal.TProgressbar",
            troughcolor="#F3F0D7",
            background="#F2C94C",
            bordercolor="#D4A017",
            lightcolor="#F7D96B",
            darkcolor="#C99812",
        )

    def _build_ui(self) -> None:
        """建立 GUI 元件。"""

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
        ttk.Label(status_frame, text="音量：").pack(side=tk.LEFT, padx=(16, 6))
        self.audio_level_bar = ttk.Progressbar(
            status_frame,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=100,
            variable=self.audio_level_var,
            length=180,
            style="AudioLevel.Horizontal.TProgressbar",
        )
        self.audio_level_bar.pack(side=tk.LEFT)

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
        """重新掃描可用輸入裝置，更新下拉選單。"""

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
        """清空共享狀態與畫面上的字幕。"""

        self.state.clear()
        self.status_var.set("已清空")
        self.audio_level_var.set(0.0)
        self.rendered_draft = ""
        self.rendered_final_lines = []
        self._replace_text(self.draft_text, "")
        self._replace_text(self.final_text, "")

    def start_engine(self) -> None:
        """驗證參數後，以背景執行緒載入模型並啟動 AudioCoordinator。"""

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
        self.rendered_draft = ""
        self.rendered_final_lines = []
        # 每次重新開始都先清空畫面，避免殘留上一輪字幕。
        self._replace_text(self.draft_text, "")
        self._replace_text(self.final_text, "")

        def runner() -> None:
            # 模型載入可能很慢，因此固定在背景執行緒進行，避免 GUI 卡死。
            try:
                self.state.set_status("載入 VAD…")
                vad_model, get_speech_timestamps = self.model_cache.get_vad()
                self.state.set_status("載入即時草稿模型…")
                draft_pipe = self.model_cache.get_draft_pipe(draft_model)
                self.state.set_status("載入 Breeze-ASR-26…")
                breeze_pipe = self.model_cache.get_breeze_pipe(breeze_model)
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
        """停止目前的錄音流程。"""

        if self.coordinator is not None:
            self.coordinator.stop()
            self.coordinator = None
        self._restore_stopped_buttons()

    def _restore_stopped_buttons(self) -> None:
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)

    def poll_state(self) -> None:
        """定期從共享狀態抓快照，將背景結果同步到 GUI。"""

        snap = self.state.snapshot()
        self.status_var.set(snap["status_text"])
        self.audio_level_var.set(snap["audio_level"])
        self._sync_draft_text(snap["draft"])
        self._sync_final_text(snap["final_lines"])
        if not snap["running"] and self.coordinator is None:
            self._restore_stopped_buttons()
        self.root.after(200, self.poll_state)

    def _replace_text(self, widget: ScrolledText, text: str) -> None:
        """完全重寫文字框內容。

        僅在清空或無法增量更新時使用。
        """

        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        if text:
            widget.insert(tk.END, text)
            widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _sync_draft_text(self, text: str) -> None:
        """只在草稿內容實際改變時更新畫面。"""

        clean = text.strip()
        if clean == self.rendered_draft:
            return
        self.rendered_draft = clean
        self._replace_text(self.draft_text, clean)

    def _sync_final_text(self, lines: list[str]) -> None:
        """對正式字幕做增量同步。

        常見情況是新的最終字幕一行一行附加，因此可以直接 append。
        如果偵測到前綴不一致，代表狀態被清空或重建，則退回全文重畫。
        """

        if lines == self.rendered_final_lines:
            return
        if len(lines) < len(self.rendered_final_lines) or lines[: len(self.rendered_final_lines)] != self.rendered_final_lines:
            self.rendered_final_lines = list(lines)
            self._replace_text(self.final_text, "\n".join(lines))
            return

        new_lines = lines[len(self.rendered_final_lines) :]
        if not new_lines:
            return

        self.final_text.configure(state=tk.NORMAL)
        prefix = "\n" if self.rendered_final_lines else ""
        self.final_text.insert(tk.END, prefix + "\n".join(new_lines))
        self.final_text.see(tk.END)
        self.final_text.configure(state=tk.DISABLED)
        self.rendered_final_lines = list(lines)

    def on_close(self) -> None:
        """關閉視窗前先停止背景流程。"""

        try:
            self.stop_engine()
        finally:
            self.root.destroy()


def parse_args() -> argparse.Namespace:
    """處理命令列參數。"""

    parser = argparse.ArgumentParser(description="Breeze 即時字幕 GUI")
    parser.add_argument("--input-device", type=int, default=None, help="sounddevice 輸入裝置 index")
    parser.add_argument("--draft-model", default="openai/whisper-small", help="即時草稿模型")
    parser.add_argument("--min-speech-ms", type=int, default=15, help="VAD 最小語音長度")
    parser.add_argument("--max-segment-s", type=float, default=14.0, help="單段最長秒數")
    parser.add_argument("--silence-end-ms", type=int, default=250, help="停頓多久視為一句結束")
    parser.add_argument("--draft-update-s", type=float, default=0.25, help="多久更新一次草稿")
    return parser.parse_args()


def main() -> int:
    """程式入口。"""

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
