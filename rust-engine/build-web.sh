#!/bin/bash
# Build script for Geometric Cognition Engine WebAssembly target

set -e

echo "Building Geometric Cognition Engine for Web..."

# Check for wasm-pack
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found. Installing..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Build with wasm-pack
echo "Compiling to WebAssembly..."
wasm-pack build --target web --out-dir web/pkg --features web

echo ""
echo "Build complete! Output in web/pkg/"
echo ""
echo "To run locally:"
echo "  cd web && python3 -m http.server 8080"
echo "  Then open http://localhost:8080"
echo ""
