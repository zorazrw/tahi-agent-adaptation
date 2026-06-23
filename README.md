
<div align="center">

# Empowering Humans with Personalized Lifelong Agent

[![Platform](https://img.shields.io/badge/platform-%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/zorazrw/agent-cowork/releases)

</div>

## 🚀 Quick Start

### Option 1: Download a release

👉 [Go to Releases](https://github.com/zorazrw/agent-cowork/releases)


### Option 2: Build from Source

#### Prerequisites

- [Bun](https://bun.sh/) or Node.js 22+
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated

```bash
git clone https://github.com/zorazrw/agent-cowork.git
cd agent-cowork
bun install
bun run dev
```

To build a production app:

```bash
bun run dist:mac-arm64    # macOS Apple Silicon (M1/M2/M3)
bun run dist:mac-x64      # macOS Intel
bun run dist:win          # Windows
bun run dist:linux        # Linux
```
