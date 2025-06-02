#!/bin/bash
# Research Assistant Pro - Installation Script

set -e  # Exit on error

echo "🚀 Research Assistant Pro Installation"
echo "====================================="
echo

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    echo "❌ Error: Python 3.11+ is required (found $python_version)"
    echo "Please install Python 3.11 or higher and try again."
    exit 1
fi

echo "✅ Python $python_version found"
echo

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Using existing environment."
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi
echo

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✅ pip upgraded"
echo

# Install package
echo "Installing Research Assistant Pro..."
pip install -e . --quiet
echo "✅ Package installed"
echo

# Download spaCy model
echo "Downloading spaCy language model..."
python -m spacy download en_core_web_sm --quiet 2>/dev/null || {
    echo "⚠️  Warning: Could not download spaCy model automatically."
    echo "   Please run: python -m spacy download en_core_web_sm"
}
echo "✅ spaCy model ready"
echo

# Create necessary directories
echo "Creating directories..."
mkdir -p ~/.research_assistant/vector_store
mkdir -p batch_results
echo "✅ Directories created"
echo

# Test installation
echo "Testing installation..."
if research-assistant-pro --help >/dev/null 2>&1; then
    echo "✅ Installation successful!"
else
    echo "❌ Installation test failed"
    exit 1
fi
echo

# Display next steps
echo "🎉 Installation Complete!"
echo
echo "Next steps:"
echo "1. Configure API keys (optional):"
echo "   research-assistant-pro configure"
echo
echo "2. Try a search:"
echo "   research-assistant-pro search -q 'your research topic'"
echo
echo "3. Find citations for your text:"
echo "   research-assistant-pro cite -t 'Your claim that needs citation'"
echo
echo "4. Run the interactive demo:"
echo "   python examples/cli_demo.py"
echo
echo "For more information, see ENHANCED_CLI_README.md"
echo
echo "Happy researching! 🔬📚"