"""Build the course archive from the packaged phase source and a compiled report."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
PREFIX = "CS781-CP-Hardik-Tiwari-230437"
SOURCE_SUFFIXES = {".py", ".ipynb", ".md", ".txt"}


def phase_files():
    for number in range(1, 8):
        phase = ROOT / "code" / f"phase{number}"
        if not any(phase.glob("*.ipynb")):
            raise ValueError(f"Missing entry notebook in {phase}")
        for path in sorted(phase.rglob("*")):
            relative = path.relative_to(phase)
            if path.is_symlink() or any(part.startswith(".") for part in relative.parts):
                continue
            is_config = relative.parent == Path("configs") and path.suffix == ".json"
            if path.is_file() and (path.suffix in SOURCE_SUFFIXES or is_config):
                yield path


def build(pdf, out_dir, draft):
    with pdf.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError(f"Invalid PDF header in {pdf}")
    files = list(phase_files())
    report_name = f"{PREFIX}-report{'-DRAFT' if draft else ''}.pdf"
    target = out_dir / f"{PREFIX}-report_code{'-DRAFT' if draft else ''}.zip"
    out_dir.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=f".{target.name}.", dir=out_dir, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(ROOT).as_posix())
            archive.write(pdf, f"report/{report_name}")
        with ZipFile(temporary) as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(f"Archive check failed for {corrupt}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Compiled report PDF")
    parser.add_argument("--out-dir", type=Path, default=ROOT.parent)
    parser.add_argument("--draft", action="store_true", help="Label the archive and PDF as drafts")
    args = parser.parse_args()
    print(build(args.pdf, args.out_dir, args.draft))


if __name__ == "__main__":
    main()
