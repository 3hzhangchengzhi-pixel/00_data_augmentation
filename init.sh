#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "  二手車データ Pipeline — 環境初始化"
echo "=========================================="

# 1. Python バージョンチェック (>= 3.12)
REQUIRED_MAJOR=3
REQUIRED_MINOR=12

PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+\.\d+')
MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$MAJOR" -lt "$REQUIRED_MAJOR" ] || { [ "$MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$MINOR" -lt "$REQUIRED_MINOR" ]; }; then
    echo "❌ Python >= ${REQUIRED_MAJOR}.${REQUIRED_MINOR} が必要です (現在: ${PYTHON_VERSION})"
    exit 1
fi
echo "✅ Python ${PYTHON_VERSION} 検出"

# 2. uv インストール（未インストールの場合）
if ! command -v uv &>/dev/null; then
    echo "📦 uv をインストール中..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "✅ uv $(uv --version) 検出"

# 3. 依存パッケージのインストール
echo "📦 依存パッケージをインストール中..."
uv sync --extra dev

# 4. 出力ディレクトリの作成
echo "📁 出力ディレクトリを作成中..."
mkdir -p out/raw out/cleaned out/models out/reports

# 5. 完了メッセージ
echo ""
echo "=========================================="
echo "  ✅ 環境セットアップ完了！"
echo "=========================================="
echo ""
echo "利用可能なコマンド:"
echo "  uv run pytest tests/ -v          # テスト実行"
echo "  uv run ruff check src/ tests/    # Lint チェック"
echo "  uv run ruff format src/ tests/   # コードフォーマット"
echo "  uv run python -m data_augmentation.websites.carsensor   # Carsensor 爬虫"
echo "  uv run python -m data_augmentation.websites.mobilico    # Mobilico 爬虫"
echo "  uv run python -m data_augmentation.websites.aucsupport  # AucSupport 爬虫"
echo "  uv run jupyter lab               # JupyterLab 起動"
echo ""
