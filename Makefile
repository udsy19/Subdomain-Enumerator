# Ultra-Robust Subdomain Enumerator - Makefile

.PHONY: build clean install deps test run help

# Binary name
BINARY_NAME = subdomain-scanner
PYTHON_CMD = python3

# Default target
help:
	@echo "🔍 Ultra-Robust Subdomain Enumerator - Build System"
	@echo "=================================================="
	@echo ""
	@echo "Available commands:"
	@echo "  build     - Build the TUI application"
	@echo "  deps      - Install all dependencies"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install to system PATH"
	@echo "  test      - Run tests"
	@echo "  run       - Build and run the application"
	@echo "  help      - Show this help message"
	@echo ""

# Install dependencies
deps:
	@echo "📦 Installing dependencies..."
	@go mod tidy
	@$(PYTHON_CMD) -m pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully!"

# Build the application
build: deps
	@echo "🔨 Building $(BINARY_NAME)..."
	@go build -ldflags "-s -w" -o $(BINARY_NAME) .
	@echo "✅ Build completed successfully!"

# Build and run
run: build
	@echo "🚀 Starting $(BINARY_NAME)..."
	@./$(BINARY_NAME)

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -f $(BINARY_NAME)
	@rm -f $(BINARY_NAME).exe
	@rm -rf output/*
	@echo "✅ Clean completed!"

# Install to system PATH (Linux/macOS)
install: build
	@echo "📦 Installing $(BINARY_NAME) to /usr/local/bin..."
	@sudo cp $(BINARY_NAME) /usr/local/bin/
	@echo "✅ Installation completed!"
	@echo "   You can now run '$(BINARY_NAME)' from anywhere"

# Test the application
test:
	@echo "🧪 Running tests..."
	@go test ./...
	@$(PYTHON_CMD) -m py_compile main_tui.py
	@echo "✅ Tests completed!"

# Cross-platform builds
build-all: deps
	@echo "🌍 Building for all platforms..."
	@GOOS=linux GOARCH=amd64 go build -ldflags "-s -w" -o $(BINARY_NAME)-linux-amd64 .
	@GOOS=darwin GOARCH=amd64 go build -ldflags "-s -w" -o $(BINARY_NAME)-darwin-amd64 .
	@GOOS=darwin GOARCH=arm64 go build -ldflags "-s -w" -o $(BINARY_NAME)-darwin-arm64 .
	@GOOS=windows GOARCH=amd64 go build -ldflags "-s -w" -o $(BINARY_NAME)-windows-amd64.exe .
	@echo "✅ Cross-platform builds completed!"
	@ls -la $(BINARY_NAME)-*

# Create release package
package: build-all
	@echo "📦 Creating release packages..."
	@mkdir -p releases
	@for binary in $(BINARY_NAME)-*; do \
		platform=$$(echo $$binary | cut -d'-' -f2-); \
		mkdir -p releases/$$platform; \
		cp $$binary releases/$$platform/$(BINARY_NAME)$$(echo $$binary | grep -o '\.exe$$' || echo ''); \
		cp main_tui.py releases/$$platform/; \
		cp requirements.txt releases/$$platform/; \
		cp -r wordlists releases/$$platform/; \
		cp README.md releases/$$platform/; \
		cp TECHNICAL_README.md releases/$$platform/; \
		cd releases && tar -czf subdomain-enumerator-$$platform.tar.gz $$platform && cd ..; \
	done
	@echo "✅ Release packages created in releases/ directory"