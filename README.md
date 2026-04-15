
<div align="center">

# Empowering Humans with Personalized Lifelong Agent

[![Platform](https://img.shields.io/badge/platform-%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/DevAgentForge/Claude-Cowork/releases)

</div>

## 🚀 Quick Start

### Option 1: Download a Release

👉 [Go to Releases](https://github.com/DevAgentForge/agent-cowork/releases)


### Option 2: Build from Source

#### Prerequisites

- [Bun](https://bun.sh/) or Node.js 22+
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated

bash
#### Clone the repository
git clone https://github.com/zorazrw/agent-cowork.git
cd agent-cowork

#### Install dependencies
bun install (bun run rebuild)
uv sync --project tinker-bridge

#### Run in development mode
bun run dev

#### Or build production binaries

```bash
bun run dist:mac-arm64    # macOS Apple Silicon (M1/M2/M3)
bun run dist:mac-x64      # macOS Intel
bun run dist:win          # Windows
bun run dist:linux        # Linux
```
