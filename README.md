# 特定人物モザイク（Web版 / 解析・確認）

動画は端末内だけで処理され、外部へは送信されません。

## 1. モデルを置く
`buffalo_s.zip` を取得（https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip ）し、
中の `det_500m.onnx` と `w600k_mbf.onnx` を `models/` に置きます。
※ InsightFaceのモデルは非商用ライセンスです。

## 2. GitHub → Cloudflare Pages
```
git init && git add . && git commit -m "init"
git branch -M main && git remote add origin <your-repo-url> && git push -u origin main
```
Cloudflare ダッシュボード → Workers & Pages → Pages → Git に接続 → リポジトリ選択。
フレームワーク「なし」、ビルドコマンド空、出力ディレクトリ `/`（空でも可）でデプロイ。
`_headers` により WASM マルチスレッドが有効になります（Chrome/Edge 推奨）。

## 3. 使い方
動画と本人の顔写真を選び「解析開始」。終わると全員に枠とIDが出ます（赤=モザイク対象）。
表のチェックで修正し、「結果をJSON保存」で `tracks.json` を出力。
次段階: 書き出し（WebCodecsでモザイク描画→MP4）、解析結果の保存/再開。
