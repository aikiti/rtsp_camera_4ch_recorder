# RTSP 4ch Recorder — Windows 一発ビルド手順

このフォルダには Windows 用のビルド補助ファイルが入っています。**事前に `rtsp_4ch_recorder.py` を同じフォルダに置いてください**（アプリ本体のスクリプト）。

## 手順（概要）
1. FFmpeg を入手して `bin/ffmpeg.exe` を配置  
   - 公式ダウンロード: ffmpeg.org → Windows builds（gyan.dev か BtbN）
2. `requirements.txt` と `make_exe.bat`、`bin/` を `rtsp_4ch_recorder.py` と同じフォルダに置く
3. `make_exe.bat` をダブルクリック
4. `dist/RTSP4CamRecorder/RTSP4CamRecorder.exe` が生成されます（**ポータブル実行可能**）

> 初回実行時に SmartScreen の警告が出る場合があります（未署名のため）。
> 「**詳細情報 (More info)** → **実行 (Run anyway)**」で起動できます。

## 詳細
- PyInstaller のオプションで PySide6 / cv2 の依存物を収集し、`bin/ffmpeg.exe` を同梱します。
- アプリ側は Windows では `./bin/ffmpeg.exe` を優先的に使用するよう実装済み。
- 出力先は `output/continuous/` と `output/events/`。プレ・ポストは UI で設定できます。
- PLC トリガーを使う場合は、**PLC 側の Host Station Port** とアプリの **Port** が一致している必要があります。

## フォルダ構成（例）
```
your-project/
├─ rtsp_4ch_recorder.py         ← アプリ本体（同じ階層に置く）
├─ make_exe.bat
├─ requirements.txt
├─ bin/
│   └─ ffmpeg.exe               ← ここに配置（gyan.dev/BtbN から取得）
└─ dist/
    └─ RTSP4CamRecorder/
        ├─ RTSP4CamRecorder.exe ← 生成物
        └─ bin/ffmpeg.exe       ← 同梱コピー
```

## 既知の注意点
- 一部のカメラは UDP より TCP の方が安定します。RTSP URL に `?tcp` を付けるなども有効です。
- もし起動しない/プレビューが出ない場合は、GPU ドライバ更新やカメラの RTSP 設定（Main/Sub、H.264/H.265、解像度など）をご確認ください。
- PLC 通信が不安定な場合は、同セグメント上のネットワーク輻輳・ファイアウォール・ポート設定を確認してください。

Happy recording!
