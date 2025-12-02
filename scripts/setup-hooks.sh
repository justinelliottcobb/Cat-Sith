#!/bin/bash
#
# Setup script for CatSith development environment
# Configures git hooks and checks dependencies
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🐱 Setting up CatSith development environment..."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check for required tools
echo "Checking dependencies..."

if ! command -v cargo &> /dev/null; then
    echo "Error: cargo not found. Please install Rust."
    exit 1
fi
echo -e "${GREEN}✓${NC} cargo found"

if ! command -v rustfmt &> /dev/null; then
    echo -e "${YELLOW}Installing rustfmt...${NC}"
    rustup component add rustfmt
fi
echo -e "${GREEN}✓${NC} rustfmt available"

if ! command -v cargo-clippy &> /dev/null; then
    echo -e "${YELLOW}Installing clippy...${NC}"
    rustup component add clippy
fi
echo -e "${GREEN}✓${NC} clippy available"

echo ""

# Configure git to use our hooks directory
echo "Configuring git hooks..."
cd "$PROJECT_ROOT"
git config core.hooksPath .githooks

# Make hooks executable
chmod +x .githooks/*

echo -e "${GREEN}✓${NC} Git hooks configured"
echo ""

# Verify setup
echo "Verifying setup..."
if [ "$(git config core.hooksPath)" = ".githooks" ]; then
    echo -e "${GREEN}✓${NC} Hooks path set correctly"
else
    echo "Warning: Hooks path not set"
fi

echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""
echo "Pre-commit hooks will now run:"
echo "  • cargo fmt --check"
echo "  • cargo clippy"
echo "  • cargo test"
echo ""
echo "To skip hooks temporarily (not recommended), use:"
echo "  git commit --no-verify"
