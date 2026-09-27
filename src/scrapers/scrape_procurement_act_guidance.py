from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
import re
import requests
from bs4 import BeautifulSoup

MAX_HTML_CHARS = 50_000
BASE_OUTPUT_DIR = Path("Procurement Act 2023 Guidance")

PHASES = {
    "Plan Phase": (
        "https://www.gov.uk/government/publications/"
        "procurement-act-2023-guidance-documents-plan-phase"
    ),
    "Define Phase": (
        "https://www.gov.uk/government/publications/"
        "procurement-act-2023-guidance-documents-define-phase"
    ),
    "Procure Phase": (
        "https://www.gov.uk/government/publications/"
        "procurement-act-2023-guidance-documents-procure-phase"
    ),
    "Manage Phase": (
        "https://www.gov.uk/government/publications/"
        "procurement-act-2023-guidance-documents-manage-phase"
    ),
}

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})


def clean_filename(name):
    name = unquote(name)
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def extract_document_links(page_url):
    response = session.get(page_url, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    links = []

    # Primary GOV.UK attachment/document heading pattern
    for heading in soup.select("h3"):
        link = heading.find("a", href=True)
        if not link:
            continue

        title = link.get_text(" ", strip=True)
        href = urljoin(page_url, link["href"])

        if "(PDF)" in title or "(HTML)" in title:
            links.append({
                "title": title,
                "url": href,
            })

    # Fallback in case GOV.UK markup changes
    if not links:
        for link in soup.select("main a[href]"):
            title = link.get_text(" ", strip=True)
            href = urljoin(page_url, link["href"])

            if "(PDF)" in title or "(HTML)" in title:
                links.append({
                    "title": title,
                    "url": href,
                })

    # Remove duplicates while preserving order
    seen = set()
    deduped = []

    for item in links:
        key = (item["title"], item["url"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


def extract_html_text(html):
    soup = BeautifulSoup(html, "html.parser")

    content = (
        soup.select_one(".govspeak")
        or soup.select_one(".gem-c-govspeak")
        or soup.select_one("main")
    )

    if content is None:
        return soup.get_text("\n", strip=True)

    for tag in content.select(
        "script, style, nav, form, button, noscript"
    ):
        tag.decompose()

    return content.get_text("\n", strip=True)


def process_phase(phase_name, page_url):
    output_dir = BASE_OUTPUT_DIR / phase_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(phase_name.upper())
    print("=" * 80)
    print(page_url)

    try:
        document_links = extract_document_links(page_url)
    except Exception as exc:
        print(f"FAILED TO READ PHASE PAGE: {exc}")
        return {
            "phase": phase_name,
            "pdfs": [],
            "html_saved": [],
            "oversized_html": [],
            "errors": [{
                "title": phase_name,
                "url": page_url,
                "error": str(exc),
            }],
        }

    print(f"Found {len(document_links)} document links")

    downloaded_pdfs = []
    saved_html = []
    oversized_html = []
    errors = []

    for item in document_links:
        title = item["title"]
        url = item["url"]

        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            is_pdf = (
                "(PDF)" in title
                or "application/pdf" in content_type
                or url.lower().endswith(".pdf")
            )

            if is_pdf:
                filename = clean_filename(
                    Path(urlparse(url).path).name
                )

                if not filename.lower().endswith(".pdf"):
                    filename += ".pdf"

                filepath = output_dir / filename
                filepath.write_bytes(response.content)

                downloaded_pdfs.append({
                    "title": title,
                    "url": url,
                    "path": str(filepath),
                    "bytes": len(response.content),
                })

                print(
                    f"PDF downloaded: {filepath} "
                    f"({len(response.content):,} bytes)"
                )

            else:
                text = extract_html_text(response.text)
                char_count = len(text)

                base_title = re.sub(
                    r"\s*\(HTML\)\s*$",
                    "",
                    title,
                    flags=re.I,
                )

                if char_count <= MAX_HTML_CHARS:
                    filename = clean_filename(base_title) + ".txt"
                    filepath = output_dir / filename

                    filepath.write_text(
                        text,
                        encoding="utf-8",
                    )

                    saved_html.append({
                        "title": base_title,
                        "url": url,
                        "path": str(filepath),
                        "chars": char_count,
                    })

                    print(
                        f"HTML saved: {char_count:,} chars | "
                        f"{filepath}"
                    )

                else:
                    oversized_html.append({
                        "phase": phase_name,
                        "title": base_title,
                        "url": url,
                        "chars": char_count,
                    })

                    print(
                        f"HTML TOO LARGE: "
                        f"{char_count:,} chars | "
                        f"{base_title}"
                    )

        except Exception as exc:
            errors.append({
                "phase": phase_name,
                "title": title,
                "url": url,
                "error": str(exc),
            })

            print(f"ERROR: {title} | {exc}")

    return {
        "phase": phase_name,
        "pdfs": downloaded_pdfs,
        "html_saved": saved_html,
        "oversized_html": oversized_html,
        "errors": errors,
    }


def main():
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for phase_name, page_url in PHASES.items():
        result = process_phase(phase_name, page_url)
        all_results.append(result)

    all_oversized = [
        item
        for result in all_results
        for item in result["oversized_html"]
    ]

    all_errors = [
        item
        for result in all_results
        for item in result["errors"]
    ]

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for result in all_results:
        print(
            f"{result['phase']}: "
            f"{len(result['pdfs'])} PDFs, "
            f"{len(result['html_saved'])} HTML text files, "
            f"{len(result['oversized_html'])} oversized HTML pages, "
            f"{len(result['errors'])} errors"
        )

    print(
        f"\nTotal PDFs downloaded: "
        f"{sum(len(r['pdfs']) for r in all_results)}"
    )
    print(
        f"Total HTML pages saved: "
        f"{sum(len(r['html_saved']) for r in all_results)}"
    )
    print(
        f"Total HTML pages over {MAX_HTML_CHARS:,} chars: "
        f"{len(all_oversized)}"
    )
    print(f"Total errors: {len(all_errors)}")

    report_path = BASE_OUTPUT_DIR / "oversized_html_report.txt"

    with report_path.open("w", encoding="utf-8") as f:
        if all_oversized:
            f.write(
                f"HTML pages over {MAX_HTML_CHARS:,} characters\n"
            )
            f.write("=" * 80 + "\n\n")

            for item in all_oversized:
                f.write(f"Phase: {item['phase']}\n")
                f.write(f"Title: {item['title']}\n")
                f.write(f"Characters: {item['chars']:,}\n")
                f.write(f"URL: {item['url']}\n")
                f.write("-" * 80 + "\n")
        else:
            f.write(
                f"No HTML pages exceeded "
                f"{MAX_HTML_CHARS:,} characters.\n"
            )

    if all_oversized:
        print("\nHTML PAGES OVER LIMIT:")
        for item in all_oversized:
            print(
                f"{item['phase']} | "
                f"{item['chars']:,} chars | "
                f"{item['title']}"
            )
            print(item["url"])

    if all_errors:
        error_report = BASE_OUTPUT_DIR / "errors_report.txt"

        with error_report.open("w", encoding="utf-8") as f:
            for item in all_errors:
                f.write(f"Phase: {item.get('phase', '')}\n")
                f.write(f"Title: {item.get('title', '')}\n")
                f.write(f"URL: {item.get('url', '')}\n")
                f.write(f"Error: {item.get('error', '')}\n")
                f.write("-" * 80 + "\n")

        print(f"\nErrors report: {error_report}")

    print(f"\nOversized HTML report: {report_path}")
    print(f"Output folder: {BASE_OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
