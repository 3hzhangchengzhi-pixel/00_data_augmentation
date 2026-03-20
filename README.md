# 二手車データ収集・価格予測 Pipeline

日本の中古車取引サイトからデータを収集し、機械学習モデルで価格を予測するパイプライン。

## アーキテクチャ

```
爬虫採集 → データクリーニング → 特徴量エンジニアリング → AI 価格予測モデル
```

## 対象サイト

- **Carsensor** (carsensor.net) — 日本最大の中古車情報サイト
- **Mobilico** (mobilico.jp) — 中古車取引プラットフォーム
- **AucSupport** (aucsupport.com) — オークションサポートサイト

## 技術スタック

- Python 3.12+, uv (パッケージ管理)
- requests + BeautifulSoup4 (スクレイピング)
- pandas, numpy (データ処理)
- scikit-learn, XGBoost, LightGBM (機械学習)
- MySQL + PostgreSQL (データベース)
- Metabase (BI ダッシュボード)
- pytest, ruff (テスト・コード品質)

## セットアップ

```bash
# 環境初始化
bash init.sh

# または手動:
uv sync
mkdir -p out/raw out/cleaned out/models out/reports
```

## 使い方

```bash
# テスト実行
uv run pytest tests/ -v

# Lint チェック
uv run ruff check src/ tests/

# 爬虫実行
uv run python -m data_augmentation.websites.carsensor
uv run python -m data_augmentation.websites.mobilico
uv run python -m data_augmentation.websites.aucsupport

# JupyterLab 起動
uv run jupyter lab
```

## プロジェクト構造

```
src/data_augmentation/
├── websites/     # 爬虫モジュール（3サイト対応）
├── cleaning/     # データクリーニング（標準化・重複排除）
├── features/     # 特徴量エンジニアリング
├── models/       # 機械学習モデル（訓練・評価・予測）
├── storage/      # データベース・エクスポート
└── research/     # Jupyter 研究ノートブック

tests/            # pytest テストスイート
dev/              # Docker Compose（Metabase + DB）
out/              # 出力データ
```

## 自主コーディングエージェント

本プロジェクトは Claude Agent SDK で駆動する自主コーディングエージェントを含む：

- `autonomous_agent_demo.py` — CLI エントリーポイント
- `agent.py` — エージェントセッションループ
- `client.py` — Claude SDK クライアント設定
- `security.py` — Bash コマンドセキュリティ検証

```bash
export ANTHROPIC_API_KEY='your-api-key'
pip install -r requirements.txt
python autonomous_agent_demo.py --project-dir ./data_pipeline
```

## 開発環境

```bash
# Docker で DB + Metabase を起動
cd dev && docker compose up -d
```
