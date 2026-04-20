# speech

這個專案是一個以 `Tkinter` 製作的即時語音辨識桌面工具，核心流程如下：

1. 使用 `sounddevice` 從麥克風擷取音訊
2. 使用 Silero VAD 判斷是否有語音並切分片段
3. 使用 Whisper 模型產生即時草稿字幕
4. 使用 `MediaTek-Research/Breeze-ASR-26` 對完整片段做正式精修

目前主要程式為 [realtime_breeze_gui.py](/Users/alex/PycharmProjects/speech/realtime_breeze_gui.py:1)。

## 功能特色

- 可選擇麥克風輸入裝置
- 顯示即時草稿字幕
- 顯示 Breeze 精修後的正式字幕
- 支援開始、停止、清空文字
- 模型快取：重複開始/停止時不必每次重載模型
- 背景草稿 worker：避免草稿辨識阻塞收音處理
- UI 增量更新：降低文字區反覆重繪成本

## 專案結構

- [realtime_breeze_gui.py](/Users/alex/PycharmProjects/speech/realtime_breeze_gui.py:1)：主程式，包含 GUI、音訊協調、VAD、ASR 流程
- [requirements_realtime_breeze.txt](/Users/alex/PycharmProjects/speech/requirements_realtime_breeze.txt:1)：執行所需 Python 套件
- [README_realtime_breeze_gui.md](/Users/alex/PycharmProjects/speech/README_realtime_breeze_gui.md:1)：GUI 使用說明與常見問題

## 安裝

建議使用 Python 3.11 以上，並先建立虛擬環境：

```bash
python3 -m venv asr_env
source asr_env/bin/activate
pip install --upgrade pip
pip install -r requirements_realtime_breeze.txt
```

## 執行

```bash
python realtime_breeze_gui.py
```

若你想指定預設麥克風裝置：

```bash
python realtime_breeze_gui.py --input-device 1
```

也可以覆蓋部分預設參數：

```bash
python realtime_breeze_gui.py \
  --draft-model openai/whisper-small \
  --min-speech-ms 15 \
  --silence-end-ms 250 \
  --draft-update-s 0.25 \
  --max-segment-s 14
```

## 目前預設值

程式目前的預設參數如下：

- `draft-model`: `openai/whisper-small`
- `min-speech-ms`: `15`
- `silence-end-ms`: `250`
- `draft-update-s`: `0.25`
- `max-segment-s`: `14.0`

這些值偏向較快的反應速度。如果環境雜音多，建議提高 `min-speech-ms`；如果希望一句話更快結束送出，可調低 `silence-end-ms`。

## 執行流程簡述

- `audio_callback()` 只負責把音訊塊送進 queue，避免在音訊 callback 做重工作
- `processing_worker()` 負責 VAD、片段累積與送出 draft/refine 任務
- `draft_worker()` 專門處理草稿辨識，避免拖慢收音流程
- `refine_worker()` 專門處理 Breeze 正式字幕精修
- `ModelCache` 會快取已載入的模型，減少重複啟動成本

## 注意事項

- 第一次啟動時，模型可能需要下載，時間會較久
- 在 macOS 上，請先確認 Terminal、iTerm、PyCharm 或 VSCode 已取得麥克風權限
- `Breeze-ASR-26` 較大，正式字幕延遲高於即時草稿屬正常現象

## 開發與驗證

目前最基本的語法檢查可使用：

```bash
python3 -m py_compile realtime_breeze_gui.py
```

如果你要看更完整的 GUI 操作說明，請參考 [README_realtime_breeze_gui.md](/Users/alex/PycharmProjects/speech/README_realtime_breeze_gui.md:1)。
