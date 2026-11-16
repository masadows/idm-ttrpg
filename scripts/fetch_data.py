"""Pobiera podręczniki z manifest.json (HTTP) lub otwiera przeglądarkę (Google Drive / MEGA), uzgadnia nazwy i weryfikuje SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import webbrowser
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel

console = Console()


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reconcile_by_hash(out_dir: Path, target_name: str, expected_sha: str) -> bool:
    """Sprawdza, czy plik docelowy istnieje (lub czy w katalogu leży plik o tym samym SHA-256 pod inną nazwą)."""
    dest = out_dir / target_name
    if dest.exists() and sha256sum(dest) == expected_sha:
        return True

    for pdf in out_dir.glob("*.pdf"):
        if pdf.name != target_name and sha256sum(pdf) == expected_sha:
            pdf.rename(dest)
            console.print(f"[cyan]↻[/] Zmieniono nazwę [dim]{pdf.name}[/] → [bold]{target_name}[/]")
            return True
    return False


def download_http(url: str, dest: Path) -> None:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    with httpx.stream("GET", url, follow_redirects=True, headers=headers, timeout=120.0) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=65536):
                f.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pobierz podręczniki RPG z manifest.json.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/manifest.json"),
        help="Ścieżka do pliku manifest.json (domyślnie: data/manifest.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Katalog docelowy na pliki PDF (domyślnie: katalog z manifest.json)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Nie otwieraj automatycznie przeglądarki dla linków wymagających ręcznego pobrania",
    )
    args = parser.parse_args()

    manifest_path: Path = args.manifest
    if not manifest_path.exists() and Path("manifest.json").exists():
        manifest_path = Path("manifest.json")
    if not manifest_path.exists():
        console.print(f"[red]Błąd:[/] Nie znaleziono pliku {manifest_path}")
        raise SystemExit(1)

    out_dir: Path = args.out_dir or manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = data.get("documents", [])

    manual_needed: list[dict[str, str]] = []
    failed = False

    for doc in documents:
        filename = doc["file"]
        expected_sha = doc["sha256"]
        url = doc.get("url", "")
        dest = out_dir / filename

        if reconcile_by_hash(out_dir, filename, expected_sha):
            console.print(f"[green]✓[/] [bold]{filename}[/] – gotowy (SHA-256 zgodne).")
            continue

        if any(domain in url for domain in ("mega.nz", "drive.google.com")):
            manual_needed.append(doc)
            continue

        if not url:
            console.print(f"[yellow]⚠ Brak URL dla {filename} w manifeście.[/]")
            failed = True
            continue

        console.print(f"[cyan]⬇[/] Pobieranie [bold]{filename}[/]...")
        try:
            download_http(url, dest)
        except Exception as err:  # noqa: BLE001
            console.print(f"[red]✗ Błąd pobierania {filename}:[/] {err}")
            failed = True
            continue

        if reconcile_by_hash(out_dir, filename, expected_sha):
            console.print(f"[green]✓[/] Pobrano i zweryfikowano [bold]{filename}[/].")
        else:
            actual = sha256sum(dest) if dest.exists() else "brak pliku"
            console.print(
                f"[red]✗ Niezgodna suma SHA-256 dla {filename}![/]\n"
                f"  Oczekiwano: {expected_sha}\n"
                f"  Otrzymano:  {actual}"
            )
            failed = True

    if manual_needed:
        for doc in manual_needed:
            url = doc["url"]
            dest = out_dir / doc["file"]
            console.print()
            console.print(
                Panel(
                    f"Pobierz plik ręcznie w przeglądarce i zapisz go w katalogu [bold]{out_dir}[/] jako:\n"
                    f"  [bold green]{dest}[/]\n\n"
                    f"Link: [underline cyan]{url}[/]\n"
                    f"[dim](Po pobraniu uruchom skrypt ponownie, aby zweryfikować SHA-256 i uzgodnić nazwę).[/]",
                    title=f"Wymagane pobranie ręczne: {doc['file']}",
                    border_style="yellow",
                )
            )
            if not args.no_browser:
                console.print(f"[dim]Otwieranie linku dla {doc['file']} w przeglądarce...[/]")
                webbrowser.open(url)
        raise SystemExit(1)

    if failed:
        raise SystemExit(1)

    console.print("\n[bold green]✅ Wszystkie podręczniki z manifest.json są pobrane i zweryfikowane![/]")


if __name__ == "__main__":
    main()