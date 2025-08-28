#!/bin/bash
# Run Arcana Application
source /Users/osmond/ArcanaExtreme/.venv/bin/activate

# Ensure the virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
  echo "Error: Virtual environment not activated. Exiting."
  exit 1
fi

echo "Starting Arcana Extreme AI Assistant..."
echo "===================="
echo ""
echo "🌟 Features:"
echo "  - Document Upload and Management"
echo "  - AI-Powered Chatbot"
echo "  - Presentation Generator"
echo "  - Study Guide Creator"
echo "  - Document Editor"
echo ""
echo "Opening in browser..."
streamlit run ArcanaExtreme.py