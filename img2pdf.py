#!/usr/bin/env python3
"""
img2pdf.py - A feature-rich terminal tool to convert images to PDF

Usage:
    python img2pdf.py photo.jpg
    python img2pdf.py *.jpg -o album.pdf -s a4
    python img2pdf.py images/ --grid 2x2 --page-size letter
    python img2pdf.py *.png --info
"""

import argparse
import io
import os
import re
import sys
import glob
import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# ─────────────────────────────────────────────
# Dependency checks
# ─────────────────────────────────────────────
MISSING = []

try:
    from PIL import Image, ImageOps
    from PIL.ExifTags import TAGS
except ImportError:
    MISSING.append("Pillow")

try:
    from reportlab.lib.pagesizes import A3, A4, A5, LETTER, LEGAL, TABLOID, B5
    from reportlab.lib.units import mm, cm, inch
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
except ImportError:
    MISSING.append("reportlab")

try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeRemainingColumn, MofNCompleteColumn, TaskProgressColumn
    )
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

try:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice
    HAS_INQUIRER = True
except ImportError:
    HAS_INQUIRER = False

if MISSING:
    msg = f"Missing required packages: {', '.join(MISSING)}\n"
    msg += f"Install with:  pip install {' '.join(MISSING)}"
    print(msg, file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
VERSION = "1.0.0"

SUPPORTED_FORMATS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif',
    '.webp', '.ico', '.ppm', '.pgm', '.pbm', '.pnm', '.avif',
}

PAGE_SIZES = {
    'a3': A3, 'a4': A4, 'a5': A5,
    'letter': LETTER, 'legal': LEGAL,
    'tabloid': TABLOID, 'b5': B5,
}


# ─────────────────────────────────────────────
# Terminal output helpers
# ─────────────────────────────────────────────
def print_warning(msg: str):
    if HAS_RICH:
        console.print(f"[yellow]  WARNING  [/yellow] {msg}")
    else:
        print(f"WARNING: {msg}", file=sys.stderr)


def print_error(msg: str):
    if HAS_RICH:
        console.print(f"[bold red]  ERROR    [/bold red] {msg}")
    else:
        print(f"ERROR: {msg}", file=sys.stderr)


def print_info(msg: str):
    if HAS_RICH:
        console.print(f"[cyan]  INFO     [/cyan] {msg}")
    else:
        print(msg)


def print_success(msg: str):
    if HAS_RICH:
        console.print(f"[green]  OK       [/green] {msg}")
    else:
        print(msg)


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────
def natural_sort_key(s: str) -> List:
    """Natural sort: '10.jpg' sorts after '9.jpg'."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s))]


def format_file_size(size: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def parse_page_size(size_str: str) -> Tuple[float, float]:
    """
    Parse page size. Returns (width, height) in points.
    Accepts: a4, letter, legal, a3, a5, b5, tabloid
             or custom: 210x297mm / 8.5x11in / 595x842pt
    """
    s = size_str.strip().lower()
    if s in PAGE_SIZES:
        return PAGE_SIZES[s]

    m = re.match(r'^(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|in|pt)?$', s)
    if m:
        w, h, unit = float(m.group(1)), float(m.group(2)), (m.group(3) or 'mm')
        factor = {'mm': mm, 'cm': cm, 'in': inch, 'pt': 1.0}[unit]
        return w * factor, h * factor

    raise ValueError(
        f"Unknown page size: '{size_str}'.\n"
        f"  Presets : a4, a3, a5, letter, legal, tabloid, b5\n"
        f"  Custom  : WxH[mm|cm|in|pt]  e.g. 210x297mm or 8.5x11in"
    )


def parse_margin(s: str) -> float:
    """Parse a margin value with optional unit."""
    m = re.match(r'^(\d+(?:\.\d+)?)\s*(mm|cm|in|pt)?$', s.strip())
    if not m:
        raise argparse.ArgumentTypeError(
            f"Invalid margin '{s}'. Examples: 10, 10mm, 0.5in, 20pt"
        )
    val, unit = float(m.group(1)), (m.group(2) or 'pt')
    return val * {'mm': mm, 'cm': cm, 'in': inch, 'pt': 1.0}[unit]


def parse_grid(s: str) -> Tuple[int, int]:
    """Parse 'CxR' grid spec."""
    m = re.match(r'^(\d+)\s*[x×]\s*(\d+)$', s.strip().lower())
    if not m:
        raise argparse.ArgumentTypeError(
            f"Invalid grid '{s}'. Use CxR format e.g. 2x2, 3x4"
        )
    c, r = int(m.group(1)), int(m.group(2))
    if c < 1 or r < 1:
        raise argparse.ArgumentTypeError("Grid dimensions must be >= 1")
    return c, r


# ─────────────────────────────────────────────
# Image file collection
# ─────────────────────────────────────────────
def collect_image_files(
    inputs: List[str],
    recursive: bool = False,
    sort_by: str = 'name',
    reverse: bool = False,
) -> List[Path]:
    """
    Collect image files from a mix of file paths, directories, and glob patterns.
    Preserves input order for sort_by='none', deduplicates by resolved path.
    """
    raw: List[Path] = []

    for inp in inputs:
        p = Path(inp)

        if p.is_file():
            if p.suffix.lower() in SUPPORTED_FORMATS:
                raw.append(p)
            else:
                print_warning(f"Skipping unsupported format: {p.name}")

        elif p.is_dir():
            pattern = '**/*' if recursive else '*'
            found = sorted(
                (f for f in p.glob(pattern)
                 if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS),
                key=lambda f: natural_sort_key(f.name)
            )
            if not found:
                print_warning(f"No images found in directory: {p}")
            raw.extend(found)

        else:
            # Try as glob pattern
            matched = [
                Path(g) for g in glob.glob(inp, recursive=recursive)
                if Path(g).is_file() and Path(g).suffix.lower() in SUPPORTED_FORMATS
            ]
            if not matched:
                print_warning(f"No matching image files: {inp}")
            raw.extend(matched)

    # Deduplicate preserving first occurrence
    seen: set = set()
    unique: List[Path] = []
    for f in raw:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)

    # Sort
    if sort_by == 'name':
        unique.sort(key=lambda f: natural_sort_key(f.name), reverse=reverse)
    elif sort_by == 'date':
        unique.sort(key=lambda f: f.stat().st_mtime, reverse=reverse)
    elif sort_by == 'size':
        unique.sort(key=lambda f: f.stat().st_size, reverse=reverse)
    elif sort_by == 'path':
        unique.sort(key=lambda f: natural_sort_key(str(f)), reverse=reverse)
    elif sort_by == 'none':
        if reverse:
            unique.reverse()

    return unique


# ─────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────
def get_exif_rotation(img: 'Image.Image') -> int:
    """Return the clockwise rotation degrees encoded in EXIF, or 0."""
    try:
        exif = img._getexif()
        if exif:
            for tag_id, value in exif.items():
                if TAGS.get(tag_id) == 'Orientation':
                    return {1: 0, 3: 180, 6: 270, 8: 90}.get(value, 0)
    except Exception:
        pass
    return 0


def open_and_prepare(path: Path, fix_rotation: bool, quality: int) -> io.BytesIO:
    """
    Open an image, correct EXIF rotation, flatten transparency onto white,
    convert to RGB, and return a JPEG-encoded BytesIO buffer.
    """
    with Image.open(path) as img:
        # Handle animated (GIF, APNG, WebP) — take first frame
        if getattr(img, 'is_animated', False):
            img.seek(0)
            img = img.copy()

        # Fix EXIF rotation
        if fix_rotation:
            rot = get_exif_rotation(img)
            if rot:
                img = img.rotate(rot, expand=True)

        # Flatten transparency onto white background
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            if img.mode == 'P':
                img = img.convert('RGBA')
            bg = Image.new('RGB', img.size, (255, 255, 255))
            alpha = img.split()[-1] if img.mode in ('RGBA', 'LA') else None
            if alpha:
                bg.paste(img.convert('RGB'), mask=alpha)
            else:
                bg.paste(img)
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality, optimize=True)
        size = img.size  # (width, height) in pixels

    buf.seek(0)
    return buf, size


def get_image_info(path: Path) -> Dict[str, Any]:
    """Return a dict of metadata for display purposes."""
    info: Dict[str, Any] = {}
    try:
        with Image.open(path) as img:
            info['format'] = img.format or path.suffix.upper().lstrip('.')
            info['mode'] = img.mode
            info['width'], info['height'] = img.size
            dpi = img.info.get('dpi')
            info['dpi'] = f"{dpi[0]:.0f}x{dpi[1]:.0f}" if isinstance(dpi, (tuple, list)) else 'N/A'
            info['animated'] = getattr(img, 'is_animated', False)
            info['frames'] = getattr(img, 'n_frames', 1)
            info['exif_rotation'] = get_exif_rotation(img)
        stat = path.stat()
        info['file_size'] = stat.st_size
        info['modified'] = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
    except Exception as e:
        info['error'] = str(e)
    return info


# ─────────────────────────────────────────────
# Layout calculation
# ─────────────────────────────────────────────
def calc_page_size(
    img_w: int, img_h: int,
    page_size_arg: str,
    orientation: str,
    dpi: int,
) -> Tuple[float, float]:
    """Return (page_width, page_height) in points."""
    if page_size_arg.lower() == 'fit':
        pw = img_w / dpi * 72.0
        ph = img_h / dpi * 72.0
    else:
        pw, ph = parse_page_size(page_size_arg)

    orient = orientation.lower()
    if orient == 'landscape' and pw < ph:
        pw, ph = ph, pw
    elif orient == 'portrait' and pw > ph:
        pw, ph = ph, pw
    elif orient == 'auto':
        if img_w > img_h and pw < ph:
            pw, ph = ph, pw
        elif img_h > img_w and pw > ph:
            pw, ph = ph, pw

    return pw, ph


def calc_image_placement(
    iw: int, ih: int,
    area_x: float, area_y: float,
    area_w: float, area_h: float,
    fit_mode: str,
    dpi: int,
) -> Tuple[float, float, float, float]:
    """Return (x, y, draw_w, draw_h) for placing image inside area."""
    if fit_mode == 'fit':
        scale = min(area_w / iw, area_h / ih)
        dw, dh = iw * scale, ih * scale
        dx = area_x + (area_w - dw) / 2
        dy = area_y + (area_h - dh) / 2

    elif fit_mode == 'fill':
        scale = max(area_w / iw, area_h / ih)
        dw, dh = iw * scale, ih * scale
        dx = area_x + (area_w - dw) / 2
        dy = area_y + (area_h - dh) / 2

    elif fit_mode == 'stretch':
        dx, dy, dw, dh = area_x, area_y, area_w, area_h

    elif fit_mode == 'center':
        dw = iw / dpi * 72.0
        dh = ih / dpi * 72.0
        dx = area_x + (area_w - dw) / 2
        dy = area_y + (area_h - dh) / 2

    else:  # fallback = fit
        scale = min(area_w / iw, area_h / ih)
        dw, dh = iw * scale, ih * scale
        dx = area_x + (area_w - dw) / 2
        dy = area_y + (area_h - dh) / 2

    return dx, dy, dw, dh


# ─────────────────────────────────────────────
# Progress context manager
# ─────────────────────────────────────────────
@contextmanager
def make_progress(total: int, quiet: bool):
    """Yield (update_fn) that accepts advance=1 and description=str."""
    if HAS_RICH and not quiet:
        prog = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}", justify="left"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        with prog:
            task_id = prog.add_task("[cyan]Processing...", total=total)

            def update(advance: int = 1, description: str = None):
                kwargs = {'advance': advance}
                if description is not None:
                    kwargs['description'] = description
                prog.update(task_id, **kwargs)

            yield update
    else:
        processed = [0]

        def update(advance: int = 1, description: str = None):
            processed[0] += advance
            if not quiet:
                print(f"  [{processed[0]}/{total}] {description or ''}", flush=True)

        yield update


# ─────────────────────────────────────────────
# PDF Builder
# ─────────────────────────────────────────────
class PDFBuilder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output = Path(args.output)

    # ── Public entry point ─────────────────────
    def build(self, image_files: List[Path]) -> bool:
        if not image_files:
            print_error("No images to process.")
            return False

        if self.args.grid:
            return self._build_grid(image_files)
        else:
            return self._build_single(image_files)

    # ── Single-image-per-page ──────────────────
    def _build_single(self, image_files: List[Path]) -> bool:
        args = self.args
        margin = args.margin

        # Peek at first image for initial page size
        try:
            buf, (iw, ih) = open_and_prepare(image_files[0], not args.no_rotation_fix, args.quality)
        except Exception as e:
            print_error(f"Cannot open first image ({image_files[0].name}): {e}")
            return False

        pw, ph = calc_page_size(iw, ih, args.page_size, args.orientation, args.dpi)
        c = canvas.Canvas(str(self.output), pagesize=(pw, ph))
        self._set_metadata(c)

        total_pages = len(image_files)
        skipped = 0

        with make_progress(total_pages, args.quiet) as update:
            for page_num, img_path in enumerate(image_files, 1):
                desc = f"[cyan]Processing[/cyan] [white]{img_path.name}[/white]"

                try:
                    buf, (iw, ih) = open_and_prepare(
                        img_path, not args.no_rotation_fix, args.quality
                    )
                except Exception as e:
                    print_warning(f"Skipping {img_path.name}: {e}")
                    skipped += 1
                    update(advance=1, description=f"[yellow]Skipped[/yellow] {img_path.name}")
                    continue

                pw, ph = calc_page_size(iw, ih, args.page_size, args.orientation, args.dpi)
                c.setPageSize((pw, ph))

                # White background
                if args.white_bg:
                    c.setFillColorRGB(1, 1, 1)
                    c.rect(0, 0, pw, ph, fill=1, stroke=0)

                # Place image
                dx, dy, dw, dh = calc_image_placement(
                    iw, ih,
                    margin, margin, pw - 2 * margin, ph - 2 * margin,
                    args.fit_mode, args.dpi,
                )
                ir = ImageReader(buf)
                c.drawImage(ir, dx, dy, dw, dh, preserveAspectRatio=False)

                # Filename label
                if args.label:
                    c.setFont("Helvetica", 7)
                    c.setFillColorRGB(0.35, 0.35, 0.35)
                    c.drawString(margin + 3, margin + 3, img_path.name)

                # Page numbers
                if args.page_numbers:
                    c.setFont("Helvetica", 8)
                    c.setFillColorRGB(0.5, 0.5, 0.5)
                    c.drawRightString(pw - margin - 3, margin + 3, f"{page_num} / {total_pages}")

                # Border
                if args.border:
                    c.setStrokeColorRGB(0.7, 0.7, 0.7)
                    c.setLineWidth(0.5)
                    c.rect(margin, margin, pw - 2 * margin, ph - 2 * margin)

                c.showPage()
                update(advance=1, description=desc)

        c.save()

        if skipped and not args.quiet:
            print_warning(f"{skipped} image(s) were skipped due to errors.")
        return True

    # ── Grid layout ────────────────────────────
    def _build_grid(self, image_files: List[Path]) -> bool:
        args = self.args
        margin = args.margin
        cols, rows = args.grid
        images_per_page = cols * rows
        padding = args.grid_padding

        # Grid always uses a fixed page size (fit → A4)
        page_size_str = args.page_size if args.page_size.lower() != 'fit' else 'a4'
        pw, ph = parse_page_size(page_size_str)

        orient = args.orientation.lower()
        if orient == 'landscape' and pw < ph:
            pw, ph = ph, pw
        elif orient == 'portrait' and pw > ph:
            pw, ph = ph, pw
        elif orient == 'auto' and cols > rows and pw < ph:
            pw, ph = ph, pw

        cell_w = (pw - 2 * margin - (cols - 1) * padding) / cols
        cell_h = (ph - 2 * margin - (rows - 1) * padding) / rows

        c = canvas.Canvas(str(self.output), pagesize=(pw, ph))
        self._set_metadata(c)

        total_pages = (len(image_files) + images_per_page - 1) // images_per_page
        skipped = 0

        with make_progress(len(image_files), args.quiet) as update:
            for page_idx in range(total_pages):
                page_imgs = image_files[page_idx * images_per_page:(page_idx + 1) * images_per_page]

                if args.white_bg:
                    c.setFillColorRGB(1, 1, 1)
                    c.rect(0, 0, pw, ph, fill=1, stroke=0)

                for slot, img_path in enumerate(page_imgs):
                    col = slot % cols
                    row = slot // cols

                    # reportlab y origin is bottom-left
                    cell_x = margin + col * (cell_w + padding)
                    cell_y = ph - margin - (row + 1) * cell_h - row * padding

                    try:
                        buf, (iw, ih) = open_and_prepare(
                            img_path, not args.no_rotation_fix, args.quality
                        )
                    except Exception as e:
                        print_warning(f"Skipping {img_path.name}: {e}")
                        skipped += 1
                        update(advance=1, description=f"[yellow]Skipped[/yellow] {img_path.name}")
                        continue

                    dx, dy, dw, dh = calc_image_placement(
                        iw, ih, cell_x, cell_y, cell_w, cell_h,
                        args.fit_mode, args.dpi,
                    )
                    ir = ImageReader(buf)
                    c.drawImage(ir, dx, dy, dw, dh, preserveAspectRatio=False)

                    if args.label:
                        c.setFont("Helvetica", 6)
                        c.setFillColorRGB(0.4, 0.4, 0.4)
                        label = img_path.name
                        if len(label) > 38:
                            label = label[:35] + '...'
                        c.drawString(cell_x + 2, cell_y + 2, label)

                    if args.border:
                        c.setStrokeColorRGB(0.75, 0.75, 0.75)
                        c.setLineWidth(0.5)
                        c.rect(cell_x, cell_y, cell_w, cell_h)

                    update(advance=1, description=f"[cyan]Processing[/cyan] {img_path.name}")

                if args.page_numbers:
                    c.setFont("Helvetica", 8)
                    c.setFillColorRGB(0.5, 0.5, 0.5)
                    c.drawCentredString(pw / 2, margin / 2 + 3, f"{page_idx + 1} / {total_pages}")

                c.showPage()

        c.save()
        if skipped and not args.quiet:
            print_warning(f"{skipped} image(s) were skipped.")
        return True

    # ── Helpers ────────────────────────────────
    def _set_metadata(self, c: 'canvas.Canvas'):
        args = self.args
        if args.title:
            c.setTitle(args.title)
        if args.author:
            c.setAuthor(args.author)
        if args.subject:
            c.setSubject(args.subject)
        if args.keywords:
            c.setKeywords(args.keywords)
        c.setCreator(f"img2pdf v{VERSION}")


# ─────────────────────────────────────────────
# Password protection (optional dependency)
# ─────────────────────────────────────────────
def apply_password(pdf_path: Path, password: str) -> bool:
    """Encrypt the PDF. Requires pypdf (pip install pypdf)."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter
        except ImportError:
            print_warning(
                "Password protection requires 'pypdf'.\n"
                "         Install with:  pip install pypdf"
            )
            return False

    try:
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.encrypt(password)
        tmp = pdf_path.with_suffix('.tmp.pdf')
        with open(tmp, 'wb') as f:
            writer.write(f)
        tmp.replace(pdf_path)
        return True
    except Exception as e:
        print_error(f"Failed to encrypt PDF: {e}")
        return False


# ─────────────────────────────────────────────
# Info / dry-run display
# ─────────────────────────────────────────────
def show_info(image_files: List[Path], args: argparse.Namespace):
    """Print detailed information about the queued images and settings."""
    if HAS_RICH:
        # Images table
        tbl = Table(
            title=f"[bold]Images to Convert[/bold]  ({len(image_files)} file{'s' if len(image_files) != 1 else ''})",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            row_styles=["", "dim"],
        )
        tbl.add_column("#", style="dim", width=4, justify="right")
        tbl.add_column("Filename", style="white", max_width=40, no_wrap=True)
        tbl.add_column("Format", style="yellow", width=7)
        tbl.add_column("Dimensions", style="green", width=12)
        tbl.add_column("File Size", style="blue", width=10, justify="right")
        tbl.add_column("DPI", style="magenta", width=10)
        tbl.add_column("Rotation", style="cyan", width=8, justify="right")
        tbl.add_column("Modified", style="dim", width=17)

        total_bytes = 0
        for i, f in enumerate(image_files, 1):
            info = get_image_info(f)
            if 'error' in info:
                tbl.add_row(str(i), f.name, "[red]ERR[/red]", info['error'], "", "", "", "")
            else:
                total_bytes += info['file_size']
                rot = f"[yellow]{info['exif_rotation']}°[/yellow]" if info['exif_rotation'] else "0°"
                tbl.add_row(
                    str(i),
                    f.name,
                    info.get('format', '?'),
                    f"{info['width']} × {info['height']}",
                    format_file_size(info['file_size']),
                    info.get('dpi', 'N/A'),
                    rot,
                    info.get('modified', ''),
                )

        console.print()
        console.print(tbl)

        # Summary
        console.print(f"\n  Total input size : [bold]{format_file_size(total_bytes)}[/bold]")
        console.print(f"  Output file      : [bold cyan]{args.output}[/bold cyan]")

        # Settings table
        stbl = Table(
            title="[bold]Conversion Settings[/bold]",
            show_header=False,
            box=None,
            padding=(0, 2),
            border_style="dim",
        )
        stbl.add_column("Key", style="cyan", width=18)
        stbl.add_column("Value", style="white")

        rows_data = [
            ("Page size", args.page_size),
            ("Orientation", args.orientation),
            ("Fit mode", args.fit_mode),
            ("Margin", f"{args.margin:.1f} pt  ({args.margin / mm:.1f} mm)"),
            ("JPEG quality", str(args.quality)),
            ("DPI", str(args.dpi)),
            ("Rotation fix", "disabled" if args.no_rotation_fix else "enabled (EXIF)"),
            ("White bg", "yes" if args.white_bg else "no"),
            ("Sort by", f"{args.sort}" + (" (reversed)" if args.reverse else "")),
        ]
        if args.grid:
            rows_data.append(("Grid layout", f"{args.grid[0]} × {args.grid[1]}  ({args.grid[0]*args.grid[1]} images/page)"))
        if args.title:
            rows_data.append(("Title", args.title))
        if args.author:
            rows_data.append(("Author", args.author))
        if args.password:
            rows_data.append(("Password", "[yellow]set (hidden)[/yellow]"))

        console.print()
        for k, v in rows_data:
            stbl.add_row(k, v)
        console.print(stbl)
        console.print()

    else:
        print(f"\nImages to convert ({len(image_files)}):")
        print("-" * 70)
        for i, f in enumerate(image_files, 1):
            info = get_image_info(f)
            if 'error' not in info:
                print(f"  {i:3}. {f.name:<40} {info['width']}x{info['height']}  {format_file_size(info['file_size'])}")
            else:
                print(f"  {i:3}. {f.name:<40} ERROR: {info['error']}")
        print(f"\nOutput: {args.output}")
        print(f"Page:   {args.page_size} | Orientation: {args.orientation} | Fit: {args.fit_mode} | Quality: {args.quality}")


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='img2pdf',
        description='Convert images to PDF - feature-rich terminal tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
INTERACTIVE MODE (run with no arguments)
  %(prog)s
      Scans the current folder, shows a checkbox list (arrow keys +
      spacebar to select, Enter to confirm), then asks for output name.
      Use -r to also scan subfolders:  %(prog)s -r

EXAMPLES
  Basic
    %(prog)s photo.jpg
    %(prog)s scan1.png scan2.png scan3.png -o document.pdf

  Multiple files / directories / globs
    %(prog)s *.jpg -o album.pdf
    %(prog)s photos/ -o album.pdf -r
    %(prog)s "shots/**/*.jpg" -r -o out.pdf

  Page size & orientation
    %(prog)s *.jpg -s a4                 # A4 (auto-orient)
    %(prog)s *.jpg -s a4 -O portrait     # force portrait
    %(prog)s *.jpg -s letter -O landscape
    %(prog)s *.jpg -s 210x297mm          # custom W x H
    %(prog)s *.jpg -s fit                # page = image size (default)

  Image fitting on the page
    %(prog)s *.jpg --fit-mode fit        # letterbox (default)
    %(prog)s *.jpg --fit-mode fill       # fill page, may crop edges
    %(prog)s *.jpg --fit-mode stretch    # stretch to fill exactly
    %(prog)s *.jpg --fit-mode center     # center at native DPI size

  Grid layout (multiple images per page)
    %(prog)s *.jpg --grid 2x2
    %(prog)s *.jpg --grid 3x4 -s a4 -O landscape

  Sorting
    %(prog)s *.jpg --sort date --reverse    # newest first
    %(prog)s *.jpg --sort size              # smallest first
    %(prog)s img1.jpg img2.jpg --sort none  # keep input order

  PDF metadata & password
    %(prog)s *.jpg --title "Holiday 2024" --author "Alice"
    %(prog)s *.jpg --password "s3cr3t"      # requires: pip install pypdf

  Decorations
    %(prog)s *.jpg --label --page-numbers --border

  Margins & JPEG quality
    %(prog)s *.jpg -m 10mm -q 95

  Preview without converting (dry run)
    %(prog)s *.jpg --info
        """,
    )

    # ── Input / Output ──────────────────────────
    io = parser.add_argument_group('Input / Output')
    io.add_argument(
        'images', nargs='*', metavar='IMAGE',
        help='Image files, directories, or glob patterns. Omit to launch the interactive picker.',
    )
    io.add_argument(
        '-o', '--output', default='output.pdf', metavar='FILE',
        help='Output PDF path (default: output.pdf)',
    )
    io.add_argument(
        '-r', '--recursive', action='store_true',
        help='Recurse into subdirectories',
    )

    # ── Page settings ───────────────────────────
    pg = parser.add_argument_group('Page Settings')
    pg.add_argument(
        '-s', '--page-size', default='fit', metavar='SIZE',
        help=(
            'Page size:  a4 | a3 | a5 | letter | legal | tabloid | b5\n'
            '            fit (default - page matches image)\n'
            '            WxH[mm|cm|in|pt]  e.g. 210x297mm'
        ),
    )
    pg.add_argument(
        '-O', '--orientation', default='auto', metavar='ORIENT',
        choices=['portrait', 'landscape', 'auto'],
        help='Orientation: portrait | landscape | auto (default: auto)',
    )
    pg.add_argument(
        '-m', '--margin', default='0', metavar='MARGIN',
        type=parse_margin,
        help='Page margin. Examples: 10  10mm  0.5in  20pt  (default: 0)',
    )

    # ── Image settings ──────────────────────────
    img = parser.add_argument_group('Image Settings')
    img.add_argument(
        '--fit-mode', default='fit', metavar='MODE',
        choices=['fit', 'fill', 'stretch', 'center'],
        help='fit (default) | fill | stretch | center',
    )
    img.add_argument(
        '-q', '--quality', default=85, type=int, metavar='1-100',
        help='JPEG compression quality 1-100 (default: 85)',
    )
    img.add_argument(
        '--dpi', default=96, type=int, metavar='N',
        help='DPI used for pixel-to-point conversion (default: 96)',
    )
    img.add_argument(
        '--no-rotation-fix', action='store_true',
        help='Disable automatic EXIF rotation correction',
    )
    img.add_argument(
        '--no-white-bg', action='store_true',
        help='Disable white background (transparent images shown as-is)',
    )

    # ── Grid ────────────────────────────────────
    grid = parser.add_argument_group('Grid Layout')
    grid.add_argument(
        '--grid', type=parse_grid, metavar='CxR',
        help='Place images in a CxR grid (e.g. 2x2, 3x4)',
    )
    grid.add_argument(
        '--grid-padding', default=5.0, type=float, metavar='PT',
        help='Gap between grid cells in points (default: 5)',
    )

    # ── Decorations ─────────────────────────────
    deco = parser.add_argument_group('Decorations')
    deco.add_argument('--label', action='store_true',
                      help='Print filename below each image')
    deco.add_argument('--page-numbers', action='store_true',
                      help='Add page numbers')
    deco.add_argument('--border', action='store_true',
                      help='Draw a thin border around each image area')

    # ── Metadata ────────────────────────────────
    meta = parser.add_argument_group('PDF Metadata')
    meta.add_argument('--title', metavar='TEXT', help='Document title')
    meta.add_argument('--author', metavar='TEXT', help='Document author')
    meta.add_argument('--subject', metavar='TEXT', help='Document subject')
    meta.add_argument('--keywords', metavar='TEXT', help='Keywords (comma-separated)')
    meta.add_argument('--password', metavar='PASS',
                      help='Encrypt PDF with password (requires: pip install pypdf)')

    # ── Sorting ─────────────────────────────────
    sort = parser.add_argument_group('Sorting')
    sort.add_argument(
        '--sort', default='name', metavar='BY',
        choices=['name', 'date', 'size', 'path', 'none'],
        help='Sort by: name (default) | date | size | path | none',
    )
    sort.add_argument('--reverse', action='store_true',
                      help='Reverse the sort order')

    # ── Control ─────────────────────────────────
    ctrl = parser.add_argument_group('Control')
    ctrl.add_argument('--info', action='store_true',
                      help='Show image details and settings, then exit (dry run)')
    ctrl.add_argument('-v', '--verbose', action='store_true',
                      help='Verbose output')
    ctrl.add_argument('--quiet', action='store_true',
                      help='Suppress all non-error output')
    ctrl.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')

    return parser


# ─────────────────────────────────────────────
# Interactive picker
# ─────────────────────────────────────────────
def _parse_selection(raw: str, total: int) -> List[int]:
    """
    Parse a selection string like '1,3-5,7' or 'all' into a list of
    0-based indices. Returns an empty list on invalid input.
    """
    raw = raw.strip().lower()
    if raw in ('a', 'all'):
        return list(range(total))

    indices: List[int] = []
    for part in raw.split(','):
        part = part.strip()
        m = re.match(r'^(\d+)-(\d+)$', part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            indices.extend(range(lo - 1, hi))  # 1-based → 0-based
        elif part.isdigit():
            indices.append(int(part) - 1)
        else:
            return []  # invalid token
    # Filter valid range
    return [i for i in indices if 0 <= i < total]


# ─────────────────────────────────────────────
# Folder scanner & feature menu (interactive)
# ─────────────────────────────────────────────
CSV_EXTENSIONS = {'.csv'}
EXCEL_EXTENSIONS = {'.xlsx', '.xls'}


def scan_script_directory() -> List[Dict[str, Any]]:
    """
    Scan all immediate subdirectories of the script's directory and
    return a list of dicts with file-type statistics.
    """
    script_dir = Path(__file__).parent.resolve()
    folders: List[Dict[str, Any]] = []

    for entry in sorted(script_dir.iterdir(), key=lambda e: e.name.lower()):
        if not entry.is_dir() or entry.name.startswith('.') or entry.name == '__pycache__':
            continue
        stats: Dict[str, Any] = {
            'name': entry.name,
            'path': entry,
            'images': 0, 'csv': 0, 'excel': 0, 'other': 0,
        }
        try:
            for f in entry.iterdir():
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                if ext in SUPPORTED_FORMATS:
                    stats['images'] += 1
                elif ext in CSV_EXTENSIONS:
                    stats['csv'] += 1
                elif ext in EXCEL_EXTENSIONS:
                    stats['excel'] += 1
                else:
                    stats['other'] += 1
        except PermissionError:
            print_warning(f"Cannot read folder: {entry.name}")
            continue
        folders.append(stats)

    return folders


def display_folder_table(folders: List[Dict[str, Any]]) -> None:
    """Display a table of folders and their detected file types."""
    if HAS_RICH:
        tbl = Table(
            title="[bold]Folders in Script Directory[/bold]",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        tbl.add_column("#", style="dim", width=4, justify="right")
        tbl.add_column("Folder", style="white", min_width=20, no_wrap=True)
        tbl.add_column("Images", style="green", width=8, justify="right")
        tbl.add_column("CSV", style="yellow", width=6, justify="right")
        tbl.add_column("Excel", style="blue", width=7, justify="right")
        tbl.add_column("Other", style="dim", width=7, justify="right")

        for i, folder in enumerate(folders, 1):
            if folder['images'] > 0:
                name = f"[bold green]{folder['name']}[/bold green]"
            else:
                name = f"[dim]{folder['name']}[/dim]"
            img_str = str(folder['images']) if folder['images'] > 0 else "-"
            csv_str = str(folder['csv']) if folder['csv'] > 0 else "-"
            xls_str = str(folder['excel']) if folder['excel'] > 0 else "-"
            oth_str = str(folder['other']) if folder['other'] > 0 else "-"
            tbl.add_row(str(i), name, img_str, csv_str, xls_str, oth_str)

        console.print()
        console.print(tbl)
        console.print()
    else:
        print(f"\nFolders found ({len(folders)}):")
        print(f"  {'#':>3}  {'Folder':<30} {'Images':>7} {'CSV':>5} {'Excel':>6} {'Other':>6}")
        print("  " + "-" * 60)
        for i, folder in enumerate(folders, 1):
            marker = " *" if folder['images'] > 0 else ""
            print(f"  {i:3}  {folder['name']:<30} {folder['images']:>7} "
                  f"{folder['csv']:>5} {folder['excel']:>6} {folder['other']:>6}{marker}")
        print()


def select_folder(folders: List[Dict[str, Any]]) -> Path:
    """
    Let the user select a folder with images.
    Returns the selected folder Path.
    """
    script_dir = Path(__file__).parent.resolve()
    folders_with_images = [f for f in folders if f['images'] > 0]

    if not folders_with_images:
        print_warning("No subfolders with images found. Using script directory.")
        return script_dir

    if HAS_INQUIRER:
        try:
            choices = []
            for f in folders_with_images:
                label = f"{f['name']:<30}  ({f['images']} image{'s' if f['images'] != 1 else ''})"
                choices.append(Choice(value=f['path'], name=label))
            choices.append(Choice(value=script_dir, name="[Script root directory]"))

            selected = inquirer.select(
                message="Select a folder to scan for images",
                choices=choices,
                cycle=True,
            ).execute()
            return selected
        except Exception:
            pass  # Fall through to plain text picker

    # Plain text fallback
    print("Folders with images:")
    for i, f in enumerate(folders_with_images, 1):
        print(f"  {i}. {f['name']} ({f['images']} images)")
    print(f"  0. [Script root directory]")

    raw = input("\nSelect folder number: ").strip()
    if raw == '0' or raw == '':
        return script_dir
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(folders_with_images):
            return folders_with_images[idx]['path']
    except ValueError:
        pass
    print_warning("Invalid selection, using script directory.")
    return script_dir


def feature_menu(args: argparse.Namespace) -> bool:
    """
    Display an interactive feature configuration menu with alphabet keys.
    Returns True to proceed with conversion, False to quit.
    Modifies args in-place.
    """

    def _get_features():
        """Build feature list with current values."""
        return [
            ('a', 'Page Size',      args.page_size),
            ('b', 'Orientation',    args.orientation),
            ('c', 'Fit Mode',       args.fit_mode),
            ('d', 'Quality',        str(args.quality)),
            ('e', 'DPI',            str(args.dpi)),
            ('f', 'Margin',         f"{args.margin:.1f} pt"),
            ('g', 'Grid Layout',    f"{args.grid[0]}x{args.grid[1]}" if args.grid else "None"),
            ('h', 'Password',       "Set" if args.password else "Not set"),
            ('i', 'Title',          args.title or "Not set"),
            ('j', 'Author',         args.author or "Not set"),
            ('k', 'Labels',         "On" if args.label else "Off"),
            ('l', 'Page Numbers',   "On" if args.page_numbers else "Off"),
            ('m', 'Borders',        "On" if args.border else "Off"),
            ('n', 'Sort By',        args.sort + (" (reversed)" if args.reverse else "")),
            ('o', 'Rotation Fix',   "Off" if args.no_rotation_fix else "On"),
            ('p', 'White BG',       "On" if args.white_bg else "Off"),
        ]

    def _render_menu(features):
        """Render the menu using Rich or plain text."""
        if HAS_RICH:
            tbl = Table(
                title="[bold]Feature Configuration[/bold]",
                show_header=True,
                header_style="bold cyan",
                border_style="dim",
                padding=(0, 1),
            )
            tbl.add_column("Key", style="bold yellow", width=5, justify="center")
            tbl.add_column("Feature", style="white", min_width=16)
            tbl.add_column("Current Value", style="green", min_width=20)

            for key, label, value in features:
                tbl.add_row(f"\\[{key}]", label, value)

            console.print()
            console.print(tbl)
            console.print()
            console.print(
                "  [bold green]\\[s][/bold green] START conversion    "
                "[bold red]\\[q][/bold red] QUIT"
            )
            console.print()
        else:
            print("\n--- Feature Configuration ---")
            for key, label, value in features:
                print(f"  [{key}] {label:<18} : {value}")
            print(f"\n  [s] START conversion    [q] QUIT\n")

    # ── Handlers ──────────────────────────────────

    def _handle_page_size():
        options = ['fit', 'a4', 'a3', 'a5', 'letter', 'legal', 'tabloid', 'b5']
        if HAS_INQUIRER:
            try:
                result = inquirer.select(
                    message="Select page size",
                    choices=options + [Choice(value='_custom', name='Custom (WxH)')],
                ).execute()
                if result == '_custom':
                    custom = inquirer.text(
                        message="Enter custom size (e.g., 210x297mm):"
                    ).execute()
                    try:
                        parse_page_size(custom)
                        args.page_size = custom
                    except ValueError as e:
                        print_error(str(e))
                else:
                    args.page_size = result
                return
            except Exception:
                pass  # Fall through to plain text
        print(f"  Options: {', '.join(options)}, or WxH[mm|cm|in|pt]")
        raw = input("  Page size: ").strip()
        if raw:
            try:
                if raw.lower() != 'fit':
                    parse_page_size(raw)
                args.page_size = raw
            except ValueError as e:
                print_error(str(e))

    def _handle_orientation():
        options = ['auto', 'portrait', 'landscape']
        idx = options.index(args.orientation) if args.orientation in options else 0
        args.orientation = options[(idx + 1) % len(options)]
        print_info(f"Orientation set to: {args.orientation}")

    def _handle_fit_mode():
        options = ['fit', 'fill', 'stretch', 'center']
        idx = options.index(args.fit_mode) if args.fit_mode in options else 0
        args.fit_mode = options[(idx + 1) % len(options)]
        print_info(f"Fit mode set to: {args.fit_mode}")

    def _handle_quality():
        if HAS_INQUIRER:
            try:
                val = inquirer.number(
                    message="JPEG quality (1-100):",
                    default=args.quality,
                    min_allowed=1,
                    max_allowed=100,
                ).execute()
                args.quality = int(val)
                return
            except Exception:
                pass
        raw = input(f"  Quality (1-100) [{args.quality}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= 100:
            args.quality = int(raw)

    def _handle_dpi():
        if HAS_INQUIRER:
            try:
                val = inquirer.number(
                    message="DPI:",
                    default=args.dpi,
                    min_allowed=1,
                ).execute()
                args.dpi = int(val)
                return
            except Exception:
                pass
        raw = input(f"  DPI [{args.dpi}]: ").strip()
        if raw.isdigit() and int(raw) > 0:
            args.dpi = int(raw)

    def _handle_margin():
        if HAS_INQUIRER:
            try:
                raw = inquirer.text(
                    message="Margin (e.g., 10mm, 0.5in, 20pt):",
                    default="0",
                ).execute()
            except Exception:
                raw = input("  Margin (e.g., 10mm, 0.5in, 20pt): ").strip()
        else:
            raw = input("  Margin (e.g., 10mm, 0.5in, 20pt): ").strip()
        if raw:
            try:
                args.margin = parse_margin(raw)
                print_info(f"Margin set to: {args.margin:.1f} pt")
            except argparse.ArgumentTypeError as e:
                print_error(str(e))

    def _handle_grid():
        if HAS_INQUIRER:
            try:
                raw = inquirer.text(
                    message="Grid layout (CxR, e.g., 2x2) or 'none' to disable:",
                    default="none",
                ).execute()
            except Exception:
                raw = input("  Grid layout (CxR or 'none'): ").strip()
        else:
            raw = input("  Grid layout (CxR or 'none'): ").strip()
        if raw.lower() in ('none', 'n', ''):
            args.grid = None
            print_info("Grid disabled.")
        else:
            try:
                args.grid = parse_grid(raw)
                print_info(f"Grid set to: {args.grid[0]}x{args.grid[1]}")
            except argparse.ArgumentTypeError as e:
                print_error(str(e))

    def _handle_password():
        if HAS_INQUIRER:
            try:
                pw = inquirer.secret(message="Password (blank to remove):").execute()
            except Exception:
                import getpass
                pw = getpass.getpass("  Password (blank to remove): ")
        else:
            import getpass
            pw = getpass.getpass("  Password (blank to remove): ")
        args.password = pw if pw.strip() else None
        print_info("Password " + ("set." if args.password else "removed."))

    def _handle_title():
        if HAS_INQUIRER:
            try:
                val = inquirer.text(
                    message="Title:", default=args.title or ""
                ).execute()
            except Exception:
                val = input(f"  Title [{args.title or ''}]: ").strip()
        else:
            val = input(f"  Title [{args.title or ''}]: ").strip()
        args.title = val if val.strip() else None

    def _handle_author():
        if HAS_INQUIRER:
            try:
                val = inquirer.text(
                    message="Author:", default=args.author or ""
                ).execute()
            except Exception:
                val = input(f"  Author [{args.author or ''}]: ").strip()
        else:
            val = input(f"  Author [{args.author or ''}]: ").strip()
        args.author = val if val.strip() else None

    def _toggle_label():
        args.label = not args.label
        print_info(f"Labels: {'On' if args.label else 'Off'}")

    def _toggle_page_numbers():
        args.page_numbers = not args.page_numbers
        print_info(f"Page numbers: {'On' if args.page_numbers else 'Off'}")

    def _toggle_border():
        args.border = not args.border
        print_info(f"Borders: {'On' if args.border else 'Off'}")

    def _handle_sort():
        options = ['name', 'date', 'size', 'path', 'none']
        idx = options.index(args.sort) if args.sort in options else 0
        args.sort = options[(idx + 1) % len(options)]
        print_info(f"Sort by: {args.sort}")

    def _toggle_rotation_fix():
        args.no_rotation_fix = not args.no_rotation_fix
        print_info(f"Rotation fix: {'Off' if args.no_rotation_fix else 'On'}")

    def _toggle_white_bg():
        args.white_bg = not args.white_bg
        print_info(f"White background: {'On' if args.white_bg else 'Off'}")

    handlers = {
        'a': _handle_page_size,
        'b': _handle_orientation,
        'c': _handle_fit_mode,
        'd': _handle_quality,
        'e': _handle_dpi,
        'f': _handle_margin,
        'g': _handle_grid,
        'h': _handle_password,
        'i': _handle_title,
        'j': _handle_author,
        'k': _toggle_label,
        'l': _toggle_page_numbers,
        'm': _toggle_border,
        'n': _handle_sort,
        'o': _toggle_rotation_fix,
        'p': _toggle_white_bg,
    }

    # ── Main loop ─────────────────────────────────
    while True:
        features = _get_features()
        _render_menu(features)

        choice = input("  Enter option: ").strip().lower()

        if choice == 's':
            return True
        elif choice == 'q':
            return False
        elif choice in handlers:
            handlers[choice]()
        else:
            print_warning(f"Unknown option '{choice}'. Use a-p, s, or q.")


def interactive_mode(args: argparse.Namespace) -> List[Path]:
    """
    Enhanced interactive mode:
    1. Scan script directory for folders and show file-type summary
    2. Let user pick a folder with images
    3. Show image picker
    4. Show alphabet-keyed feature configuration menu
    5. Ask for output filename
    6. Return selected Path list (modifies args in-place)
    """

    # ── Step 1: Folder scanning ───────────────────────────────────────
    if not args.quiet:
        print_info("Scanning script directory for folders...")

    folders = scan_script_directory()

    if folders:
        display_folder_table(folders)
        selected_folder = select_folder(folders)
    else:
        selected_folder = Path(__file__).parent.resolve()
        if not args.quiet:
            print_info("No subfolders found. Using script directory.")

    # ── Step 2: Collect images from selected folder ───────────────────
    if not args.quiet:
        print_info(f"Scanning '{selected_folder.name}' for images...")

    all_files = collect_image_files(
        [str(selected_folder)],
        recursive=args.recursive,
        sort_by=args.sort,
        reverse=args.reverse,
    )

    if not all_files:
        print_error(
            f"No supported image files found in '{selected_folder.name}'.\n"
            "         Tip: use -r to search recursively, or pass paths as arguments."
        )
        sys.exit(1)

    # ── Step 3: Image picker ──────────────────────────────────────────
    use_inquirer_picker = False
    if HAS_INQUIRER:
        try:
            choices = []
            for f in all_files:
                try:
                    info = get_image_info(f)
                    if 'error' not in info:
                        label = (
                            f"{f.name:<36}  "
                            f"{info['width']:>5}x{info['height']:<5}  "
                            f"{format_file_size(info['file_size']):>9}"
                        )
                    else:
                        label = f"{f.name}  [error reading]"
                except Exception:
                    label = f.name
                choices.append(Choice(value=f, name=label))

            if HAS_RICH:
                console.print()
                console.print(
                    f"  [bold cyan]{len(all_files)}[/bold cyan] image(s) found in "
                    f"[dim]{selected_folder}[/dim]\n"
                )

            selected: List[Path] = inquirer.checkbox(
                message="Select images to include in the PDF",
                choices=choices,
                cycle=True,
                instruction="(Space=toggle  Up/Down=navigate  /=search  Enter=confirm)",
                transformer=lambda result: f"{len(result)} image(s) selected",
                validate=lambda result: len(result) > 0,
                invalid_message="Select at least one image.",
            ).execute()

            if not selected:
                print_warning("No images selected. Exiting.")
                sys.exit(0)

            use_inquirer_picker = True
        except Exception:
            pass  # Fall through to plain text picker

    if not use_inquirer_picker:
        # Fallback: plain text picker
        if HAS_RICH:
            tbl = Table(show_header=True, header_style="bold cyan", border_style="dim")
            tbl.add_column("#", style="dim", width=4, justify="right")
            tbl.add_column("Filename", style="white", max_width=38, no_wrap=True)
            tbl.add_column("Dimensions", style="green", width=12)
            tbl.add_column("Size", style="blue", width=10, justify="right")
            for i, f in enumerate(all_files, 1):
                info = get_image_info(f)
                if 'error' not in info:
                    tbl.add_row(
                        str(i), f.name,
                        f"{info['width']}x{info['height']}",
                        format_file_size(info['file_size']),
                    )
                else:
                    tbl.add_row(str(i), f.name, "[red]error[/red]", "")
            console.print()
            console.print(tbl)
        else:
            print(f"\nFound {len(all_files)} image(s):")
            for i, f in enumerate(all_files, 1):
                print(f"  {i:3}. {f.name}")

        print()
        raw = input("Select images  [all / 1,2,3 / 1-5 / q to quit]: ").strip()
        if raw.lower() in ('q', 'quit', ''):
            print("Aborted.")
            sys.exit(0)

        indices = _parse_selection(raw, len(all_files))
        if not indices:
            print_error(f"Invalid selection: '{raw}'. Use: all, 1,2,3 or 1-5")
            sys.exit(1)

        selected = [all_files[i] for i in sorted(set(indices))]

        if HAS_RICH:
            console.print(f"  [green]Selected {len(selected)} image(s).[/green]")
        else:
            print(f"Selected {len(selected)} image(s).")

    # ── Step 4: Feature configuration menu ────────────────────────────
    original_sort = args.sort
    original_reverse = args.reverse

    proceed = feature_menu(args)
    if not proceed:
        print_info("Cancelled.")
        sys.exit(0)

    # Re-sort selected images if sort settings changed
    if args.sort != original_sort or args.reverse != original_reverse:
        if args.sort == 'name':
            selected.sort(key=lambda f: natural_sort_key(f.name), reverse=args.reverse)
        elif args.sort == 'date':
            selected.sort(key=lambda f: f.stat().st_mtime, reverse=args.reverse)
        elif args.sort == 'size':
            selected.sort(key=lambda f: f.stat().st_size, reverse=args.reverse)
        elif args.sort == 'path':
            selected.sort(key=lambda f: natural_sort_key(str(f)), reverse=args.reverse)
        elif args.sort == 'none' and args.reverse:
            selected.reverse()

    # ── Step 5: Output filename ───────────────────────────────────────
    default_out = "output.pdf"
    if HAS_INQUIRER:
        try:
            output_name: str = inquirer.text(
                message="Output PDF filename:",
                default=default_out,
                validate=lambda v: len(v.strip()) > 0,
                invalid_message="Please enter a filename.",
            ).execute()
            args.output = output_name.strip() or default_out
        except Exception:
            raw_out = input(f"Output PDF filename [{default_out}]: ").strip()
            args.output = raw_out or default_out
    else:
        raw_out = input(f"Output PDF filename [{default_out}]: ").strip()
        args.output = raw_out or default_out

    return selected


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = create_parser()
    args = parser.parse_args()

    # Derive white_bg from flag
    args.white_bg = not args.no_white_bg

    # Validate quality
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")

    # Pretty banner
    if not args.quiet and HAS_RICH:
        console.print(Panel.fit(
            f"[bold cyan]img2pdf[/bold cyan] [dim]v{VERSION}[/dim]  -  Image to PDF Converter",
            border_style="cyan",
            padding=(0, 2),
        ))

    # Validate page-size early (better error message)
    if args.page_size.lower() != 'fit':
        try:
            parse_page_size(args.page_size)
        except ValueError as e:
            parser.error(str(e))

    # Collect image files — or launch interactive picker
    if not args.images:
        image_files = interactive_mode(args)
    else:
        if not args.quiet:
            print_info("Scanning for images...")
        image_files = collect_image_files(
            args.images,
            recursive=args.recursive,
            sort_by=args.sort,
            reverse=args.reverse,
        )

    if not image_files:
        print_error("No supported image files found.")
        sys.exit(1)

    if not args.quiet:
        print_info(f"Found {len(image_files)} image(s)")

    # Info / dry-run mode
    if args.info:
        show_info(image_files, args)
        sys.exit(0)

    # Ensure .pdf extension
    output = Path(args.output)
    if output.suffix.lower() != '.pdf':
        output = output.with_suffix('.pdf')
        args.output = str(output)

    # Warn if overwriting
    if output.exists() and not args.quiet:
        print_warning(f"Output file already exists and will be overwritten: {output}")

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Build PDF
    start = datetime.datetime.now()
    builder = PDFBuilder(args)
    ok = builder.build(image_files)

    if not ok:
        print_error("PDF creation failed.")
        sys.exit(1)

    # Optional password protection
    if args.password:
        if not args.quiet:
            print_info("Encrypting PDF...")
        if apply_password(output, args.password):
            if not args.quiet:
                print_success("Password applied.")
        else:
            print_warning("PDF saved without password protection.")

    elapsed = (datetime.datetime.now() - start).total_seconds()

    if not args.quiet:
        out_size = output.stat().st_size if output.exists() else 0
        if args.grid:
            ipp = args.grid[0] * args.grid[1]
            pages = (len(image_files) + ipp - 1) // ipp
        else:
            pages = len(image_files)

        if HAS_RICH:
            console.print()
            console.print(Panel(
                f"[bold green]PDF created successfully![/bold green]\n\n"
                f"  [bold]Output :[/bold] {output}\n"
                f"  [bold]Size   :[/bold] {format_file_size(out_size)}\n"
                f"  [bold]Pages  :[/bold] {pages}\n"
                f"  [bold]Images :[/bold] {len(image_files)}\n"
                f"  [bold]Time   :[/bold] {elapsed:.2f}s",
                border_style="green",
                title="[bold green]Done[/bold green]",
                padding=(0, 2),
            ))
        else:
            print(f"\nDone! -> {output}  ({format_file_size(out_size)}, {pages} pages, {elapsed:.2f}s)")


if __name__ == '__main__':
    main()
