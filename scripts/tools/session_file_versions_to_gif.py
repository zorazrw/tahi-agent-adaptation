#!/usr/bin/env python3
"""
Build animated GIFs from unique HTML/PNG snapshots in an exported session.

Discovers ``.html`` / ``.png`` workspace files automatically (skips ``.py`` and other
text artifacts). Writes only the GIF path(s) you request—intermediate frames use a temp dir.

Requires: ``playwright`` + ``playwright install chromium`` (HTML), ``ffmpeg`` (GIF).

Examples:
  python scripts/tools/session_file_versions_to_gif.py \\
    -j sessions/export.json -s <uuid> -o chart.gif

  # One GIF per visual file (default names: <file-stem>.gif in cwd)
  python scripts/tools/session_file_versions_to_gif.py -j sessions/export.json -s <uuid>

Supports ``trajectory`` and weight-format ``task_units`` / ``agent_trajectories``.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_tools = Path(__file__).resolve().parent
if str(_tools) not in sys.path:
    sys.path.insert(0, str(_tools))

from rate_file_versions import load_sessions, resolve_session, session_trajectory  # noqa: E402

_VISUAL_SUFFIXES = frozenset({".html", ".htm", ".png"})
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\n\r]+$")


def get_file_snapshot(file_field: Any, filename: str) -> str | None:
    if file_field is None:
        return None
    if isinstance(file_field, dict):
        val = file_field.get(filename)
        return val if isinstance(val, str) else None
    if isinstance(file_field, list):
        for item in file_field:
            if isinstance(item, dict) and item.get("path") == filename:
                c = item.get("content")
                return c if isinstance(c, str) else None
    return None


def iter_paths_in_file_field(file_field: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(file_field, dict):
        paths = [p for p in file_field if isinstance(p, str)]
    elif isinstance(file_field, list):
        for item in file_field:
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                paths.append(item["path"])
    return paths


def is_visual_path(path: str) -> bool:
    return Path(path).suffix.lower() in _VISUAL_SUFFIXES


def discover_visual_paths(traj: list[dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    for step in traj:
        if not isinstance(step, dict):
            continue
        env = step.get("environment")
        if not isinstance(env, dict):
            continue
        for path in iter_paths_in_file_field(env.get("file")):
            if is_visual_path(path):
                found.add(path)
    return sorted(found)


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


def _image_placeholder(content: str) -> bool:
    t = content.strip()
    return not t or t.startswith("[") or "Read image file" in t


def _looks_like_base64(s: str, *, min_len: int = 64) -> bool:
    t = s.strip()
    return len(t) >= min_len and _BASE64_RE.fullmatch(t) is not None


def decode_png_bytes(content: str) -> bytes | None:
    if _image_placeholder(content):
        return None
    raw = content.strip()
    if raw.startswith("\x89PNG"):
        return raw.encode("latin-1")
    if _looks_like_base64(raw):
        try:
            return base64.b64decode(raw, validate=True)
        except (ValueError, binascii.Error):
            pass
    try:
        return base64.b64decode(raw)
    except Exception:
        return None


def usable_versions(traj: list[dict[str, Any]], path: str) -> list[str]:
    raw = collect_unique_versions(traj, path)
    ext = Path(path).suffix.lower()
    if ext == ".png":
        out: list[str] = []
        for content in raw:
            if decode_png_bytes(content) is not None:
                out.append(content)
        return out
    return [c for c in raw if c.strip() and not _image_placeholder(c)]


def _safe_segment(name: str) -> str:
    base = Path(name).name
    return re.sub(r"[^\w.\-]+", "_", base) or "file"


def write_html_frames(versions: list[str], html_dir: Path) -> list[Path]:
    html_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, html in enumerate(versions):
        p = html_dir / f"frame_{i:03d}.html"
        p.write_text(html, encoding="utf-8")
        paths.append(p)
    return paths


def write_png_frames(versions: list[str], frames_dir: Path) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, content in enumerate(versions):
        data = decode_png_bytes(content)
        if data is None:
            continue
        p = frames_dir / f"frame_{i:03d}.png"
        p.write_bytes(data)
        paths.append(p)
    return paths


def step_label(index: int, *, one_based: bool, pad: int, prefix: str) -> str:
    n = index + (1 if one_based else 0)
    body = str(n).zfill(pad) if pad > 0 else str(n)
    return f"{prefix}: {body}"


def render_html_frames_playwright(
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
) -> list[Path]:
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
    position: 'fixed', top: '18px', right: '22px',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif",
    fontSize: '22px', fontWeight: '600', color: '#111827', zIndex: '2147483647',
    pointerEvents: 'none', padding: '8px 12px', borderRadius: '8px',
    background: 'rgba(255,255,255,0.92)', boxShadow: '0 1px 3px rgba(0,0,0,0.12)',
  });
  document.body.appendChild(el);
}
"""
    out_paths: list[Path] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": viewport_w, "height": viewport_h})
        for i, f in enumerate(html_paths):
            page.goto(f.resolve().as_uri(), wait_until="networkidle")
            page.wait_for_timeout(settle_ms)
            if step_overlay:
                page.evaluate(
                    overlay_js,
                    step_label(i, one_based=one_based, pad=step_pad, prefix=step_prefix),
                )
                page.wait_for_timeout(100)
            out = frames_dir / f"frame_{i:03d}.png"
            page.screenshot(path=str(out), full_page=True)
            out_paths.append(out)
        browser.close()
    return out_paths


def build_gif_ffmpeg(frames_dir: Path, gif_path: Path, *, fps: float, scale_width: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH; install ffmpeg to build the GIF.")

    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise SystemExit("No frames to encode into GIF.")

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "frame_%03d.png")
    vf = (
        f"fps={fps},scale={scale_width}:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=single[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
    )
    subprocess.run(
        [
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
        ],
        check=True,
    )


def build_gif_for_file(
    path: str,
    versions: list[str],
    gif_path: Path,
    *,
    viewport_w: int,
    viewport_h: int,
    fps: float,
    scale_width: int,
    step_overlay: bool,
    one_based: bool,
    step_pad: int,
    step_prefix: str,
    settle_ms: int,
) -> None:
    ext = Path(path).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="session-gif-") as tmp:
        work = Path(tmp)
        frames_dir = work / "frames"

        if ext in (".html", ".htm"):
            html_paths = write_html_frames(versions, work / "html")
            try:
                render_html_frames_playwright(
                    html_paths,
                    frames_dir,
                    viewport_w=viewport_w,
                    viewport_h=viewport_h,
                    step_overlay=step_overlay,
                    one_based=one_based,
                    step_pad=step_pad,
                    step_prefix=step_prefix,
                    settle_ms=settle_ms,
                )
            except ImportError as exc:
                raise SystemExit(
                    "Playwright is required for HTML GIFs. Install with:\n"
                    "  pip install playwright\n"
                    "  playwright install chromium",
                ) from exc
        elif ext == ".png":
            png_paths = write_png_frames(versions, frames_dir)
            if not png_paths:
                raise SystemExit(f"No decodable PNG snapshots for {path!r}")
        else:
            raise SystemExit(f"Not a visual file: {path!r}")

        build_gif_ffmpeg(frames_dir, gif_path, fps=fps, scale_width=scale_width)


def resolve_gif_targets(
    visual_paths: list[str],
    *,
    output: Path | None,
    file_filter: str | None,
) -> list[tuple[str, Path]]:
    if file_filter:
        if not is_visual_path(file_filter):
            raise SystemExit(f"--file must be .html or .png, got {file_filter!r}")
        visual_paths = [file_filter]

    if not visual_paths:
        raise SystemExit("No .html or .png paths found in session file snapshots")

    if output is not None:
        out = output.resolve()
        if out.suffix.lower() != ".gif":
            out = out.with_suffix(".gif")
        if len(visual_paths) == 1:
            return [(visual_paths[0], out)]
        parent, stem = out.parent, out.stem
        return [(p, parent / f"{stem}_{_safe_segment(Path(p).stem)}.gif") for p in visual_paths]

    return [(p, Path.cwd() / f"{Path(p).stem}.gif") for p in visual_paths]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-j", "--json", type=Path, required=True, help="Exported session JSON")
    ap.add_argument("-s", "--session", default=None, help="Session uuid (required if JSON has multiple)")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH.gif",
        help="Output GIF file. With multiple visual artifacts, writes stem_<file>.gif siblings.",
    )
    ap.add_argument(
        "-f",
        "--file",
        default=None,
        metavar="PATH",
        help="Optional: only this workspace file (.html or .png). Default: all visual files with snapshots.",
    )
    ap.add_argument("--fps", type=float, default=0.75, help="GIF frame rate (default: 0.75)")
    ap.add_argument("--scale", type=int, default=1280, metavar="W", help="GIF width in pixels (default: 1280)")
    ap.add_argument("--viewport", default="1920x1200", metavar="WxH", help="Chromium viewport for HTML")
    ap.add_argument("--settle-ms", type=int, default=400, help="Wait after HTML load before screenshot")
    ap.add_argument("--no-step-label", action="store_true", help="Omit step number overlay on frames")
    ap.add_argument("--step-prefix", default="Step")
    ap.add_argument("--zero-based-step", action="store_true")
    ap.add_argument("--step-pad", type=int, default=2)
    args = ap.parse_args()

    if not args.json.is_file():
        raise SystemExit(f"JSON file not found: {args.json}")

    session = resolve_session(load_sessions(args.json), args.session)
    name = session.get("name", "")
    traj = session_trajectory(session)
    if not traj:
        raise SystemExit("Session has no trajectory or task_units with file snapshots")

    visual_paths = discover_visual_paths(traj)
    targets = resolve_gif_targets(visual_paths, output=args.output, file_filter=args.file)

    vw, vh = args.viewport.lower().split("x", 1)
    viewport_w, viewport_h = int(vw), int(vh)

    wrote: list[Path] = []
    skipped: list[str] = []
    for path, gif_path in targets:
        versions = usable_versions(traj, path)
        if not versions:
            skipped.append(path)
            continue
        build_gif_for_file(
            path,
            versions,
            gif_path,
            viewport_w=viewport_w,
            viewport_h=viewport_h,
            fps=args.fps,
            scale_width=args.scale,
            step_overlay=not args.no_step_label,
            one_based=not args.zero_based_step,
            step_pad=max(0, args.step_pad),
            step_prefix=args.step_prefix,
            settle_ms=max(0, args.settle_ms),
        )
        wrote.append(gif_path.resolve())

    uid = session.get("uuid") or args.session or ""
    print(f"Session: {uid}" + (f" — {name}" if name else ""))
    for p in wrote:
        print(f"GIF: {p}")
    if skipped:
        print("Skipped (no usable snapshots): " + ", ".join(skipped), file=sys.stderr)
    if not wrote:
        raise SystemExit(
            "No GIFs written. Weight exports often lack embedded .png bytes; "
            "try an HTML artifact or a session with image content in environment.file."
        )


if __name__ == "__main__":
    main()
