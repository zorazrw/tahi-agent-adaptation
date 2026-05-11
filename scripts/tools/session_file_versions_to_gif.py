#!/usr/bin/env python3
"""
Collect unique snapshots of a workspace file from an exported task session JSON and
optionally render them to an animated GIF.

``environment.file`` may be either:
  - a dict: ``{ "path.html": "<content or null>", ... }``
  - a list: ``[ { "path": "...", "content": "..." }, ... ]``

Only non-null string contents are collected. Duplicates are dropped while preserving
first-seen order.

Requires for GIF rendering:
  - Python package ``playwright`` and ``playwright install chromium``
  - ``ffmpeg`` on PATH

Examples:
  python scripts/session_file_versions_to_gif.py \\
    --json scripts/out.json \\
    --session 6178b6e7-53b4-4d2f-a02e-cb0e3b21cbd3 \\
    --file webarena_trend.html

  # HTML snapshots only (no Playwright/ffmpeg):
  python scripts/session_file_versions_to_gif.py -j out.json -s <uuid> -f chart.html --html-only
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_sessions(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw]
    raise ValueError("JSON root must be an array of sessions or a single session object")


def _find_session(sessions: list[dict[str, Any]], session_id: str) -> dict[str, Any]:
    for s in sessions:
        if s.get("uuid") == session_id:
            return s
    raise SystemExit(f"No session with uuid {session_id!r} in {len(sessions)} session(s)")


def get_file_snapshot(file_field: Any, filename: str) -> str | None:
    """Return file text for ``filename`` from ``environment.file``, or None."""
    if file_field is None:
        return None
    if isinstance(file_field, dict):
        val = file_field.get(filename)
        return val if isinstance(val, str) else None
    if isinstance(file_field, list):
        for item in file_field:
            if not isinstance(item, dict):
                continue
            if item.get("path") == filename:
                c = item.get("content")
                return c if isinstance(c, str) else None
    return None


def collect_unique_versions(trajectory: list[dict[str, Any]], filename: str) -> list[str]:
    versions: list[str] = []
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        env = step.get("environment")
        if not isinstance(env, dict):
            continue
        content = get_file_snapshot(env.get("file"), filename)
        if content is None:
            continue
        if content not in versions:
            versions.append(content)
    return versions


def _safe_segment(name: str) -> str:
    base = Path(name).name
    return re.sub(r"[^\w.\-]+", "_", base) or "file"


def write_html_versions(versions: list[str], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, html in enumerate(versions):
        p = out_dir / f"{i}.html"
        p.write_text(html, encoding="utf-8")
        paths.append(p)
    return paths


def step_label(
    index: int,
    *,
    one_based: bool,
    pad: int,
    prefix: str,
) -> str:
    n = index + (1 if one_based else 0)
    body = str(n).zfill(pad) if pad > 0 else str(n)
    return f"{prefix}: {body}"


def render_frames_playwright(
    html_paths: list[Path],
    frames_dir: Path,
    *,
    viewport_w: int,
    viewport_h: int,
    step_overlay: bool,
    one_based: bool,
    step_pad: int,
    step_prefix: str,
    settle_ms: int,
) -> None:
    from playwright.sync_api import sync_playwright

    frames_dir.mkdir(parents=True, exist_ok=True)
    for p in frames_dir.glob("*.png"):
        p.unlink()

    overlay_js = """
(label) => {
  const el = document.createElement('div');
  el.setAttribute('data-session-gif-step-overlay', '1');
  el.textContent = label;
  Object.assign(el.style, {
    position: 'fixed',
    top: '18px',
    right: '22px',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif",
    fontSize: '22px',
    fontWeight: '600',
    color: '#111827',
    letterSpacing: '0.02em',
    zIndex: '2147483647',
    pointerEvents: 'none',
    padding: '8px 12px',
    borderRadius: '8px',
    background: 'rgba(255,255,255,0.92)',
    boxShadow: '0 1px 3px rgba(0,0,0,0.12)',
  });
  document.body.appendChild(el);
}
"""

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": viewport_w, "height": viewport_h})
        for i, f in enumerate(html_paths):
            page.goto(f.resolve().as_uri(), wait_until="networkidle")
            page.wait_for_timeout(settle_ms)
            if step_overlay:
                lab = step_label(
                    i,
                    one_based=one_based,
                    pad=step_pad,
                    prefix=step_prefix,
                )
                page.evaluate(overlay_js, lab)
                page.wait_for_timeout(100)
            page.screenshot(path=str(frames_dir / f"frame_{i:03d}.png"), full_page=True)
        browser.close()


def build_gif_ffmpeg(
    frames_dir: Path,
    gif_path: Path,
    *,
    fps: float,
    scale_width: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH; install ffmpeg to build the GIF.")

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "frame_%03d.png")
    vf = (
        f"fps={fps},scale={scale_width}:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=single[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
    )
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-lavfi",
        vf,
        str(gif_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export unique file snapshots from a session trajectory and optionally build a GIF.",
    )
    ap.add_argument(
        "-j",
        "--json",
        type=Path,
        default=Path("scripts/out.json"),
        help="Exported sessions JSON (array of sessions or one object). Default: scripts/out.json",
    )
    ap.add_argument(
        "-s",
        "--session",
        required=True,
        metavar="UUID",
        help="Session uuid (see ``uuid`` in the export)",
    )
    ap.add_argument(
        "-f",
        "--file",
        required=True,
        metavar="PATH",
        help="Workspace path key to collect (e.g. webarena_trend.html)",
    )
    ap.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: ./session-html-gif/<short-uuid>/<safe-file-name>/)",
    )
    ap.add_argument(
        "--gif-name",
        default=None,
        metavar="NAME",
        help="GIF filename inside out-dir (default: <file-stem>.gif)",
    )
    ap.add_argument(
        "--html-only",
        action="store_true",
        help="Only write numbered HTML files; skip Playwright and ffmpeg",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=0.75,
        help="Frames per second for the GIF (default: 0.75)",
    )
    ap.add_argument(
        "--scale",
        type=int,
        default=1280,
        metavar="W",
        help="Output GIF width in pixels; height scales (default: 1280)",
    )
    ap.add_argument(
        "--viewport",
        default="1920x1200",
        metavar="WxH",
        help="Chromium viewport for screenshots (default: 1920x1200)",
    )
    ap.add_argument(
        "--settle-ms",
        type=int,
        default=400,
        metavar="MS",
        help="Wait after load before screenshot (default: 400)",
    )
    ap.add_argument(
        "--no-step-label",
        action="store_true",
        help="Do not draw the top-right step label on frames",
    )
    ap.add_argument(
        "--step-prefix",
        default="Step",
        help="Label prefix before the number (default: Step)",
    )
    ap.add_argument(
        "--zero-based-step",
        action="store_true",
        help="Label numbers start at 0 (default: first label is 1)",
    )
    ap.add_argument(
        "--step-pad",
        type=int,
        default=2,
        metavar="N",
        help="Zero-pad step number to this width (0 to disable; default: 2)",
    )

    args = ap.parse_args()

    if not args.json.is_file():
        raise SystemExit(f"JSON file not found: {args.json}")

    sessions = _load_sessions(args.json)
    session = _find_session(sessions, args.session)
    name = session.get("name", "")
    traj = session.get("trajectory")
    if not isinstance(traj, list):
        raise SystemExit("Session has no trajectory list")

    versions = collect_unique_versions(traj, args.file)
    if not versions:
        raise SystemExit(
            f"No non-null snapshots for {args.file!r} in session {args.session!r}"
            + (f" ({name})" if name else ""),
        )

    short = args.session[:8]
    safe = _safe_segment(args.file)
    out_dir = args.out_dir or (Path.cwd() / "session-html-gif" / short / safe)
    out_dir = out_dir.resolve()
    gif_name = args.gif_name or f"{Path(args.file).stem}.gif"
    gif_path = out_dir / gif_name
    frames_dir = out_dir / "_frames"

    html_paths = write_html_versions(versions, out_dir)
    print(f"Session: {args.session}" + (f" — {name}" if name else ""))
    print(f"Unique snapshots: {len(versions)} → {out_dir}/{{0..{len(versions) - 1}}}.html")

    if args.html_only:
        print("Done (--html-only).")
        return

    vw, vh = args.viewport.lower().split("x", 1)
    viewport_w, viewport_h = int(vw), int(vh)

    try:
        render_frames_playwright(
            html_paths,
            frames_dir,
            viewport_w=viewport_w,
            viewport_h=viewport_h,
            step_overlay=not args.no_step_label,
            one_based=not args.zero_based_step,
            step_pad=max(0, args.step_pad),
            step_prefix=args.step_prefix,
            settle_ms=max(0, args.settle_ms),
        )
    except ImportError:
        raise SystemExit(
            "Playwright is required for GIF rendering. Install with:\n"
            "  pip install playwright\n"
            "  playwright install chromium",
        ) from None

    build_gif_ffmpeg(frames_dir, gif_path, fps=args.fps, scale_width=args.scale)
    print(f"GIF: {gif_path}")


if __name__ == "__main__":
    main()
