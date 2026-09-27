from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
from email.message import Message
import csv
import mimetypes
import re
import requests
from bs4 import BeautifulSoup

COLLECTION_URL = "https://www.gov.uk/government/collections/procurement-policy-notes"

MAX_HTML_CHARS = 50_000
OUTPUT_ROOT = Path("Procurement Policy Notes")

TARGET_SECTIONS = [
    "Procurement Act 2023 PPNs",
    "Public Contract Regulations 2015 PPNs",
    "Out of date procurement policy notes",
]

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (compatible; GOV.UK PPN document scraper/1.0)"
    )
})


def normalise_space(value):
    return re.sub(r"\s+", " ", value or "").strip()


def clean_filename(value, max_length=180):
    value = unquote(str(value))
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = normalise_space(value)
    value = value.strip(" .")
    if not value:
        value = "untitled"
    return value[:max_length].rstrip(" .")


def unique_path(path):
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    n = 2

    while True:
        candidate = parent / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def filename_from_response(response, fallback_url, fallback_title="document"):
    cd = response.headers.get("Content-Disposition", "")

    if cd:
        msg = Message()
        msg["content-disposition"] = cd
        filename = msg.get_filename()
        if filename:
            return clean_filename(filename)

    url_name = Path(urlparse(fallback_url).path).name
    if url_name and "." in url_name:
        return clean_filename(url_name)

    content_type = response.headers.get("Content-Type", "")
    mime = content_type.split(";", 1)[0].strip().lower()
    extension = mimetypes.guess_extension(mime) or ""

    if mime == "application/pdf":
        extension = ".pdf"
    elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        extension = ".docx"
    elif mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        extension = ".xlsx"
    elif mime == "application/vnd.ms-excel":
        extension = ".xls"
    elif mime == "text/csv":
        extension = ".csv"
    elif mime == "application/zip":
        extension = ".zip"

    return clean_filename(fallback_title) + extension


def extract_main_html_text(html):
    soup = BeautifulSoup(html, "html.parser")

    content = (
        soup.select_one(".govspeak")
        or soup.select_one(".gem-c-govspeak")
        or soup.select_one("article")
        or soup.select_one("main")
    )

    if content is None:
        content = soup

    # Remove obvious page chrome / interactive elements.
    for tag in content.select(
        "script, style, nav, form, button, noscript, "
        ".gem-c-feedback, .govuk-footer, .govuk-header"
    ):
        tag.decompose()

    return content.get_text("\n", strip=True)


def get_collection_sections():
    response = session.get(COLLECTION_URL, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    result = {}

    for section_name in TARGET_SECTIONS:
        heading = None

        for candidate in soup.find_all(["h2", "h3"]):
            if normalise_space(candidate.get_text(" ", strip=True)).casefold() == section_name.casefold():
                heading = candidate
                break

        if heading is None:
            raise RuntimeError(f"Could not find section heading: {section_name}")

        links = []
        seen = set()

        # Traverse everything following this heading until the next heading
        # at the same or higher level.
        heading_level = int(heading.name[1])

        for element in heading.find_all_next():
            if element is heading:
                continue

            if element.name in ("h2", "h3"):
                level = int(element.name[1])
                if level <= heading_level:
                    break

            if element.name != "a" or not element.get("href"):
                continue

            href = urljoin(COLLECTION_URL, element["href"])
            title = normalise_space(element.get_text(" ", strip=True))

            # Collection entries are GOV.UK publication / guidance / policy pages.
            parsed = urlparse(href)
            if parsed.netloc not in ("www.gov.uk", "gov.uk"):
                continue

            if not (
                parsed.path.startswith("/government/publications/")
                or parsed.path.startswith("/government/guidance/")
            ):
                continue

            key = href.split("#", 1)[0]
            if key in seen:
                continue

            seen.add(key)
            links.append({
                "title": title or Path(parsed.path).name,
                "url": key,
            })

        result[section_name] = links

    return result


def get_document_links(publication_url):
    response = session.get(publication_url, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    documents_heading = None
    for h2 in soup.find_all("h2"):
        if normalise_space(h2.get_text(" ", strip=True)).casefold() == "documents":
            documents_heading = h2
            break

    if documents_heading is None:
        return [], response.text

    links = []
    seen = set()

    for element in documents_heading.find_all_next():
        if element is documents_heading:
            continue

        if element.name == "h2":
            break

        if element.name != "a" or not element.get("href"):
            continue

        href = urljoin(publication_url, element["href"])
        title = normalise_space(element.get_text(" ", strip=True))

        parsed = urlparse(href)

        # Only links that are plausibly documents:
        # 1) publishing.service.gov.uk attachments
        # 2) GOV.UK child HTML pages belonging to a publication
        is_asset = parsed.netloc == "assets.publishing.service.gov.uk"
        is_govuk_document = (
            parsed.netloc in ("www.gov.uk", "gov.uk")
            and parsed.path.startswith("/government/publications/")
        )

        if not (is_asset or is_govuk_document):
            continue

        key = href.split("#", 1)[0]
        if key in seen:
            continue

        seen.add(key)
        links.append({
            "title": title or Path(parsed.path).name,
            "url": key,
        })

    return links, response.text


def save_html_document(section_name, ppn_title, doc_title, url, response, ppn_dir):
    text = extract_main_html_text(response.text)
    chars = len(text)

    if chars > MAX_HTML_CHARS:
        return {
            "status": "oversized_html",
            "section": section_name,
            "ppn": ppn_title,
            "title": doc_title,
            "url": url,
            "chars": chars,
            "path": "",
        }

    filename = clean_filename(doc_title)

    # Avoid trailing "(HTML)" in saved filenames.
    filename = re.sub(r"\s*\(HTML\)\s*$", "", filename, flags=re.I).strip()

    path = unique_path(ppn_dir / f"{filename}.txt")
    path.write_text(text, encoding="utf-8")

    return {
        "status": "html_saved",
        "section": section_name,
        "ppn": ppn_title,
        "title": doc_title,
        "url": url,
        "chars": chars,
        "path": str(path),
    }


def process_document(section_name, ppn_title, doc, ppn_dir):
    url = doc["url"]
    title = doc["title"]

    response = session.get(url, timeout=90, allow_redirects=True)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    final_url = response.url
    final_host = urlparse(final_url).netloc

    is_html = (
        "text/html" in content_type
        or "application/xhtml+xml" in content_type
    )

    # GOV.UK document pages should be treated as HTML text.
    if is_html and final_host in ("www.gov.uk", "gov.uk"):
        return save_html_document(
            section_name,
            ppn_title,
            title,
            final_url,
            response,
            ppn_dir,
        )

    # Everything else under Documents is downloaded as a file.
    filename = filename_from_response(response, final_url, title)
    path = unique_path(ppn_dir / filename)
    path.write_bytes(response.content)

    return {
        "status": "file_downloaded",
        "section": section_name,
        "ppn": ppn_title,
        "title": title,
        "url": final_url,
        "chars": "",
        "path": str(path),
        "bytes": len(response.content),
        "content_type": content_type,
    }


def write_csv(path, rows):
    fields = [
        "status",
        "section",
        "ppn",
        "title",
        "url",
        "chars",
        "bytes",
        "content_type",
        "path",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            writer.writerow({
                field: row.get(field, "")
                for field in fields
            })


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Collection: {COLLECTION_URL}")
    print(f"HTML character limit: {MAX_HTML_CHARS:,}")

    sections = get_collection_sections()

    all_results = []
    errors = []

    for section_name, ppns in sections.items():
        section_dir = OUTPUT_ROOT / clean_filename(section_name)
        section_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 90)
        print(section_name.upper())
        print("=" * 90)
        print(f"PPN pages found: {len(ppns)}")

        for index, ppn in enumerate(ppns, start=1):
            ppn_title = ppn["title"]
            ppn_url = ppn["url"]
            ppn_dir = section_dir / clean_filename(ppn_title)
            ppn_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[{index}/{len(ppns)}] {ppn_title}")
            print(ppn_url)

            try:
                document_links, publication_html = get_document_links(ppn_url)

                # Record source PPN page URL for traceability.
                (ppn_dir / "_source_url.txt").write_text(
                    ppn_url + "\n",
                    encoding="utf-8",
                )

                if not document_links:
                    print("  No Documents attachments found.")

                    # If a GOV.UK item itself is content rather than a publication
                    # with attachments, save its main text under the same 50k rule.
                    page_text = extract_main_html_text(publication_html)
                    chars = len(page_text)

                    if chars <= MAX_HTML_CHARS:
                        path = ppn_dir / "_page_content.txt"
                        path.write_text(page_text, encoding="utf-8")
                        all_results.append({
                            "status": "ppn_page_html_saved",
                            "section": section_name,
                            "ppn": ppn_title,
                            "title": ppn_title,
                            "url": ppn_url,
                            "chars": chars,
                            "path": str(path),
                        })
                        print(f"  PPN page text saved: {chars:,} chars")
                    else:
                        all_results.append({
                            "status": "oversized_html",
                            "section": section_name,
                            "ppn": ppn_title,
                            "title": ppn_title,
                            "url": ppn_url,
                            "chars": chars,
                        })
                        print(f"  PPN PAGE TOO LARGE: {chars:,} chars")

                    continue

                print(f"  Documents found: {len(document_links)}")

                for doc in document_links:
                    try:
                        result = process_document(
                            section_name,
                            ppn_title,
                            doc,
                            ppn_dir,
                        )
                        all_results.append(result)

                        if result["status"] == "file_downloaded":
                            print(
                                f"  Downloaded: {Path(result['path']).name} "
                                f"({result['bytes']:,} bytes)"
                            )
                        elif result["status"] == "html_saved":
                            print(
                                f"  HTML saved: {Path(result['path']).name} "
                                f"({result['chars']:,} chars)"
                            )
                        elif result["status"] == "oversized_html":
                            print(
                                f"  HTML TOO LARGE: {result['chars']:,} chars | "
                                f"{result['title']}"
                            )

                    except Exception as exc:
                        error = {
                            "status": "error",
                            "section": section_name,
                            "ppn": ppn_title,
                            "title": doc["title"],
                            "url": doc["url"],
                            "error": str(exc),
                        }
                        errors.append(error)
                        all_results.append(error)
                        print(f"  ERROR: {doc['title']} | {exc}")

            except Exception as exc:
                error = {
                    "status": "error",
                    "section": section_name,
                    "ppn": ppn_title,
                    "title": ppn_title,
                    "url": ppn_url,
                    "error": str(exc),
                }
                errors.append(error)
                all_results.append(error)
                print(f"  PPN PAGE ERROR: {exc}")

    manifest_path = OUTPUT_ROOT / "manifest.csv"
    write_csv(manifest_path, all_results)

    oversized = [
        row for row in all_results
        if row.get("status") == "oversized_html"
    ]

    oversized_report = OUTPUT_ROOT / "oversized_html_report.txt"
    with oversized_report.open("w", encoding="utf-8") as f:
        f.write(
            f"HTML documents over {MAX_HTML_CHARS:,} characters\n"
            + "=" * 90
            + "\n\n"
        )

        if not oversized:
            f.write("None.\n")
        else:
            for row in oversized:
                f.write(f"Section: {row['section']}\n")
                f.write(f"PPN: {row['ppn']}\n")
                f.write(f"Document: {row['title']}\n")
                f.write(f"Characters: {row['chars']:,}\n")
                f.write(f"URL: {row['url']}\n")
                f.write("-" * 90 + "\n")

    if errors:
        errors_path = OUTPUT_ROOT / "errors_report.txt"
        with errors_path.open("w", encoding="utf-8") as f:
            for row in errors:
                f.write(f"Section: {row['section']}\n")
                f.write(f"PPN: {row['ppn']}\n")
                f.write(f"Document: {row['title']}\n")
                f.write(f"URL: {row['url']}\n")
                f.write(f"Error: {row['error']}\n")
                f.write("-" * 90 + "\n")

    downloaded = sum(
        1 for row in all_results
        if row.get("status") == "file_downloaded"
    )
    html_saved = sum(
        1 for row in all_results
        if row.get("status") in ("html_saved", "ppn_page_html_saved")
    )

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    for section_name, ppns in sections.items():
        print(f"{section_name}: {len(ppns)} PPN pages")

    print(f"Files downloaded: {downloaded}")
    print(f"HTML text files saved: {html_saved}")
    print(f"HTML documents over {MAX_HTML_CHARS:,} chars: {len(oversized)}")
    print(f"Errors: {len(errors)}")
    print(f"Manifest: {manifest_path}")
    print(f"Oversized report: {oversized_report}")
    print(f"Output root: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
