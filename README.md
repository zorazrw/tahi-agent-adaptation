
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
```

#### 2. Install

```bash
bun install
```

`bun install` runs a **postinstall** hook that:

1. **Rebuilds native modules** — `electron-rebuild` for `better-sqlite3` (required for the Electron app).
2. **Syncs the Tinker bridge** — `uv sync --project tinker-bridge` (creates `tinker-bridge/.venv` for the optional Tinker provider).

If `uv` is not installed, step 2 is skipped with a warning; the rest of the app still works. Install [uv](https://docs.astral.sh/uv/) and run: `bun run sync:tinker-bridge`.

#### 3. Run in development

```bash
bun run dev
```

Or build production binaries

```bash
bun run dist:mac-arm64    # macOS Apple Silicon (M1/M2/M3)
bun run dist:mac-x64      # macOS Intel
bun run dist:win          # Windows
bun run dist:linux        # Linux
```
