#!/usr/bin/env bash
# scripts/start.sh — Start the RAG application
set -e

echo "🚀 Starting RAG Application..."

# Check .env
if [ ! -f .env ]; then
  echo "⚠️  .env not found — copying from .env.example"
  cp .env.example .env
  echo "📝 Edit .env and add your OPENAI_API_KEY, then re-run this script."
  exit 1
fi

# Install dependencies (if venv not active)
if [ -z "$VIRTUAL_ENV" ]; then
  echo "💡 Creating virtual environment..."
  python -m venv .venv
  source .venv/bin/activate
fi

echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Start FastAPI
echo "🔧 Starting FastAPI backend on http://localhost:8000"
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
