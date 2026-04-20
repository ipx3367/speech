# Breeze 即時字幕 GUI 版

這是桌面版 GUI，功能包含：
- 麥克風裝置選擇
- 即時草稿字幕
- Breeze-ASR-26 背景精修後的正式字幕
- 開始 / 停止 / 清空文字

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

若你想指定預設麥克風：

```bash
python realtime_breeze_gui.py --input-device 1
```

## 建議設定

- min speech ms: 60
- silence end ms: 700
- draft update s: 0.8
- max segment s: 6.0

如果你發現說話後反應偏慢，可先把 `min speech ms` 降到 30~50。
如果環境雜音多、容易誤觸發，可提高到 80。

## 常見問題

### 1. 說話沒反應
- 檢查 macOS 麥克風權限是否已開給 Terminal / iTerm / VSCode
- 確認 GUI 上方選到正確輸入裝置
- 先把 `min speech ms` 調成 30 測試

### 2. 第一次啟動很久
第一次會下載並載入模型，這是正常的。

### 3. 正式字幕很慢
Breeze-ASR-26 是較大的模型，正式字幕會比即時草稿慢一些，這是雙層架構的正常現象。
