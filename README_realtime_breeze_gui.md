# Breeze 即時字幕 GUI 版

這是桌面版的即時語音辨識工具，使用麥克風收音後，會先顯示 Whisper 產生的即時草稿，再以 `Breeze-ASR-26` 對完整片段做正式精修。

## 功能

- 麥克風輸入裝置選擇
- 即時草稿字幕
- Breeze 精修後的正式字幕
- 開始 / 停止 / 清空文字
- 模型快取，重複啟動時不必每次重載模型
- 草稿辨識與正式精修分開執行，降低收音阻塞風險

## 安裝

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

若要指定預設麥克風裝置：

```bash
python realtime_breeze_gui.py --input-device 1
```

若要自行覆蓋預設參數：

```bash
python realtime_breeze_gui.py \
  --draft-model openai/whisper-small \
  --min-speech-ms 15 \
  --silence-end-ms 250 \
  --draft-update-s 0.25 \
  --max-segment-s 14
```

## 介面說明

- 麥克風：選擇輸入裝置
- 草稿模型：選擇即時字幕使用的 Whisper 模型
- 精修模型：目前固定為 `MediaTek-Research/Breeze-ASR-26`
- `min speech ms`：VAD 判定為語音所需的最小長度
- `silence end ms`：停頓多久後，將目前一句話視為結束
- `draft update s`：草稿字幕多久更新一次
- `max segment s`：單一語音片段最長秒數，超過會強制送去精修

## 目前程式預設值

目前實作中的預設值如下：

- `draft-model`: `openai/whisper-small`
- `min speech ms`: `15`
- `silence end ms`: `250`
- `draft update s`: `0.25`
- `max segment s`: `14.0`

這組預設值偏向「即時反應較快」。如果你想調整體驗，可參考下面的建議。

## 調整建議

如果你想讓系統更靈敏、草稿更快出現：

- 將 `min speech ms` 降低到 `10 ~ 15`
- 將 `draft update s` 維持在 `0.25 ~ 0.4`

如果你想降低雜音誤觸發：

- 將 `min speech ms` 提高到 `30 ~ 80`
- 視情況將 `silence end ms` 提高到 `400 ~ 800`

如果你想讓字幕更快斷句送出：

- 將 `silence end ms` 降到 `200 ~ 300`
- 將 `max segment s` 降到 `6 ~ 10`

如果你希望正式字幕一次吃更長的上下文：

- 將 `max segment s` 提高到 `12 ~ 18`
- 代價是每段正式字幕完成的等待時間可能更長

## 執行流程

程式內部大致分成四段：

1. `audio_callback()` 只把音訊塊送進 queue
2. `processing_worker()` 做 VAD、切段、排入草稿與正式辨識
3. `draft_worker()` 專門處理草稿字幕
4. `refine_worker()` 專門處理 Breeze 正式精修

另外，模型會由 `ModelCache` 快取，因此同一輪 GUI 生命週期內，重複按開始/停止時不必每次重新載入。

## 使用建議

- 第一次啟動時，模型可能需要下載，屬正常現象
- 如果你在 macOS 上執行，請先確認 Terminal、iTerm、PyCharm 或 VSCode 已取得麥克風權限
- 即時草稿追求低延遲，品質可能略低於正式字幕
- Breeze-ASR-26 較大，正式字幕比草稿慢是正常行為

## 常見問題

### 1. 說話沒反應

- 檢查 macOS 麥克風權限是否已開給目前執行程式的應用程式
- 確認 GUI 上方選到正確輸入裝置
- 先把 `min speech ms` 調成 `10` 或 `15` 測試
- 確認目前麥克風真的有輸入，不是選到虛擬或靜音裝置

### 2. 第一次啟動很久

第一次可能會下載並載入模型，這是正常的。之後同一個 GUI session 內再次按開始，模型會優先重用快取。

### 3. 正式字幕很慢

`Breeze-ASR-26` 是較大的模型，正式字幕會比即時草稿慢一些。若你想縮短等待時間，可以降低 `max segment s` 或 `silence end ms`，讓片段更早送出。

### 4. 草稿字幕更新不夠頻繁

可以降低 `draft update s`。但這會增加草稿辨識次數，CPU / GPU 使用率也可能提高。

### 5. 字幕太容易被切成很多小段

可提高 `silence end ms`，讓系統在較長停頓後才結束一句；若環境很吵，也建議一起提高 `min speech ms`。
