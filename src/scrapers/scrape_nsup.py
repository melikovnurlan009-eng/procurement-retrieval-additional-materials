#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, sys, time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse, urldefrag
import requests
from bs4 import BeautifulSoup, NavigableString, Tag

VERSION = "1.0.0"
BASE_URL = "https://www.gov.uk/guidance/the-national-security-unit-for-procurement"
REQUESTED_URL = BASE_URL + "#national-security-debarments"
REQUESTED_FRAGMENT = "national-security-debarments"
CONTENT_API_URL = "https://www.gov.uk/api/content/guidance/the-national-security-unit-for-procurement"
SOURCE_ID = "GOVUK_NSUP"

def utc_now(): return datetime.now(timezone.utc).isoformat()
def clean_ws(s): return re.sub(r"[ \t\r\f\v]+", " ", s or "").strip()
def sha256_bytes(b): return hashlib.sha256(b).hexdigest()
def sha256_text(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()
def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

@dataclass
class FetchRecord:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    retrieved_at: str
    sha256: str
    byte_count: int

class NSUPScraper:
    def __init__(self, output_dir: Path, timeout=60, pause=0.4, section_only=False):
        self.output_dir = output_dir
        self.raw_dir = output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.timeout, self.pause, self.section_only = timeout, pause, section_only
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MastersThesisProcurementResearchBot/1.0",
            "Accept-Language": "en-GB,en;q=0.9"
        })
        self.fetch_log = []

    def fetch(self, url, accept):
        r = self.session.get(url, headers={"Accept": accept}, timeout=self.timeout, allow_redirects=True)
        r.raise_for_status()
        data = r.content
        self.fetch_log.append(asdict(FetchRecord(
            url, r.url, r.status_code, r.headers.get("content-type",""),
            utc_now(), sha256_bytes(data), len(data)
        )))
        time.sleep(self.pause)
        return data, r.url, r.headers.get("content-type","")

    def fetch_api(self):
        try:
            data, _, _ = self.fetch(CONTENT_API_URL, "application/json,*/*;q=0.5")
            (self.raw_dir/"govuk_content_api.json").write_bytes(data)
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            (self.raw_dir/"govuk_content_api_error.json").write_text(
                json.dumps({"error":str(e),"attempted_url":CONTENT_API_URL}, indent=2),
                encoding="utf-8")
            return None

    def get_body_html(self, rendered_soup, api):
        if api:
            body = (api.get("details") or {}).get("body")
            if isinstance(body, str) and clean_ws(BeautifulSoup(body,"html.parser").get_text(" ", strip=True)):
                return body, "GOVUK_CONTENT_API_DETAILS_BODY"
        main = rendered_soup.find("main")
        if not main: raise RuntimeError("Could not find GOV.UK main content")
        for c in [main.select_one(".govspeak"), main.select_one(".gem-c-govspeak"), main.select_one("article")]:
            if c and clean_ws(c.get_text(" ", strip=True)):
                return str(c), "RENDERED_HTML_MAIN_BODY"
        return str(main), "RENDERED_HTML_MAIN_FALLBACK"

    def clean_body(self, soup):
        for t in soup.find_all(["script","style","noscript","form","button","input","select","textarea"]):
            t.decompose()

    def extract_fragment(self, soup, fragment):
        target = soup.find(id=fragment)
        if target is None:
            wanted = fragment.replace("-"," ").lower()
            for h in soup.find_all(["h1","h2","h3","h4","h5","h6"]):
                if clean_ws(h.get_text(" ", strip=True)).lower() == wanted:
                    target = h
                    break
        if target is None: raise RuntimeError(f"Could not find #{fragment}")
        if target.name not in {"h1","h2","h3","h4","h5","h6"}:
            h = target.find(["h1","h2","h3","h4","h5","h6"])
            if h: target = h
        if target.name not in {"h1","h2","h3","h4","h5","h6"}:
            raise RuntimeError("Fragment did not resolve to heading")
        level = int(target.name[1])
        parts = [str(target)]
        for s in target.next_siblings:
            if isinstance(s, Tag) and s.name in {"h1","h2","h3","h4","h5","h6"} and int(s.name[1]) <= level:
                break
            parts.append(str(s))
        return BeautifulSoup("<div>"+"".join(parts)+"</div>", "html.parser")

    def nearest_heading(self, node):
        h = node.find_previous(["h1","h2","h3","h4","h5","h6"])
        return clean_ws(h.get_text(" ", strip=True)) if h else None

    def context(self, a, n=700):
        p = a.find_parent(["p","li","td","th","div"])
        return clean_ws((p or a).get_text(" ", strip=True))[:n]

    def classify_link(self, url):
        p = urlparse(url); host = p.netloc.lower(); path = p.path.lower()
        if host.endswith("legislation.gov.uk"): return "LEGISLATION"
        if host == "assets.publishing.service.gov.uk":
            if path.endswith(".pdf"): return "GOVUK_PDF_ATTACHMENT"
            if path.endswith(".odt"): return "GOVUK_ODT_ATTACHMENT"
            return "GOVUK_ASSET"
        if host == "submit.forms.service.gov.uk": return "GOVUK_FORM"
        if host == "www.gov.uk":
            if path.startswith("/guidance/"): return "GOVUK_GUIDANCE"
            if path.startswith("/government/publications/"): return "GOVUK_PUBLICATION"
            if path.startswith("/government/collections/"): return "GOVUK_COLLECTION"
            return "GOVUK_PAGE"
        if host.endswith("gov.uk"): return "OTHER_GOVUK_SERVICE"
        return "EXTERNAL"

    def replace_links(self, soup, source_url):
        out=[]; i=0
        for a in list(soup.find_all("a", href=True)):
            href = clean_ws(a.get("href"))
            if not href: continue
            i += 1
            ph = f"[[LINK_{i:04d}]]"
            abs_url = urljoin(source_url, href)
            defrag, frag = urldefrag(abs_url)
            text = clean_ws(a.get_text(" ", strip=True))
            out.append({
                "source_id": SOURCE_ID,
                "placeholder": ph,
                "occurrence": i,
                "anchor_text": text,
                "raw_href": href,
                "absolute_url": abs_url,
                "defragmented_url": defrag,
                "fragment": frag or None,
                "link_class": self.classify_link(abs_url),
                "source_heading": self.nearest_heading(a),
                "source_context": self.context(a),
                "followed": False,
                "target_document_id": None,
                "resolution_status": "UNRESOLVED",
                "edge_status": "NOT_CREATED"
            })
            a.replace_with(NavigableString((text+" " if text else "") + ph))
        return out

    def structured_text(self, soup):
        lines=[]
        for n in soup.find_all(["h1","h2","h3","h4","h5","h6","p","li","tr","blockquote"]):
            text = clean_ws(n.get_text(" ", strip=True))
            if not text: continue
            if re.fullmatch(r"h[1-6]", n.name or ""):
                lines += ["", "#"*int(n.name[1])+" "+text, ""]
            elif n.name=="li": lines.append("- "+text)
            elif n.name=="tr":
                cells=[clean_ws(c.get_text(" ",strip=True)) for c in n.find_all(["th","td"],recursive=False)]
                if any(cells): lines.append(" | ".join([c for c in cells if c]))
            elif n.name=="blockquote": lines.append("> "+text)
            else: lines += [text,""]
        return "\n".join(lines).strip()

    def headings(self, soup):
        out=[]; stack=[]
        for i,h in enumerate(soup.find_all(["h1","h2","h3","h4","h5","h6"]),1):
            lvl=int(h.name[1]); text=clean_ws(h.get_text(" ",strip=True))
            if not text: continue
            stack=[x for x in stack if x[0] < lvl]; stack.append((lvl,text))
            out.append({"source_id":SOURCE_ID,"ordinal":i,"heading_level":lvl,"heading":text,"heading_path":[x[1] for x in stack]})
        return out

    def updates(self, soup):
        out=[]
        for h in soup.find_all(["h2","h3"]):
            if "updates to this page" in clean_ws(h.get_text(" ",strip=True)).lower():
                lvl=int(h.name[1]); parts=[]
                for s in h.next_siblings:
                    if isinstance(s,Tag) and s.name in {"h1","h2","h3","h4","h5","h6"} and int(s.name[1]) <= lvl: break
                    if isinstance(s,Tag):
                        t=clean_ws(s.get_text(" ",strip=True))
                        if t: parts.append(t)
                txt=" ".join(parts)
                for m in re.finditer(r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})\s+(.*?)(?=(?:\d{1,2}\s+[A-Za-z]+\s+\d{4})|$)", txt):
                    out.append({"source_id":SOURCE_ID,"date_text":m.group(1),"note":clean_ws(m.group(2))})
                break
        return out

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw, final_url, _ = self.fetch(BASE_URL, "text/html,*/*;q=0.5")
        (self.raw_dir/"rendered_page.html").write_bytes(raw)
        api = self.fetch_api()
        rendered = BeautifulSoup(raw, "html.parser")
        body_html, method = self.get_body_html(rendered, api)
        body = BeautifulSoup(body_html, "html.parser")
        self.clean_body(body)
        if self.section_only:
            body = self.extract_fragment(body, REQUESTED_FRAGMENT)

        links = self.replace_links(body, final_url)
        text = self.structured_text(body)
        html = str(body)
        heads = self.headings(body)
        ups = self.updates(rendered)

        (self.output_dir/"content_with_link_placeholders.txt").write_text(text, encoding="utf-8")
        (self.output_dir/"content_with_link_placeholders.html").write_text(html, encoding="utf-8")
        write_jsonl(self.output_dir/"links.jsonl", links)
        write_jsonl(self.output_dir/"headings.jsonl", heads)
        write_jsonl(self.output_dir/"updates.jsonl", ups)

        meta = {
            "source_id": SOURCE_ID,
            "requested_url": REQUESTED_URL,
            "canonical_url": BASE_URL,
            "requested_fragment": REQUESTED_FRAGMENT,
            "scope": "REQUESTED_SECTION_ONLY" if self.section_only else "FULL_CURRENT_PAGE",
            "title": (api or {}).get("title") or clean_ws((rendered.find("h1") or "").get_text(" ",strip=True)),
            "source_type": "GOVUK_GUIDANCE",
            "evidence_class": "OFFICIAL_SPECIALIST_GUIDANCE",
            "authority_class": "OFFICIAL_GOVERNMENT_GUIDANCE",
            "retrieval_lane": "GUIDANCE",
            "topic": ["NATIONAL_SECURITY","DEBARMENT","EXCLUSION","TERMINATION"],
            "content_id": (api or {}).get("content_id"),
            "document_type": (api or {}).get("document_type"),
            "schema_name": (api or {}).get("schema_name"),
            "published_at": (api or {}).get("first_published_at"),
            "public_updated_at": (api or {}).get("public_updated_at"),
            "extraction_method": method,
            "document_status": "WITHDRAWN" if "this guidance was withdrawn" in clean_ws(rendered.get_text(" ",strip=True)).lower() else "CURRENT",
            "content_text_sha256": sha256_text(text),
            "content_html_sha256": sha256_text(html),
            "content_char_count": len(text),
            "link_count": len(links),
            "heading_count": len(heads),
            "update_event_count": len(ups),
            "chunking_status": "NOT_CHUNKED",
            "link_resolution_status": "NOT_FOLLOWED",
            "graph_edge_status": "NOT_CREATED"
        }
        (self.output_dir/"document.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
        manifest = {
            "scraper_version": VERSION,
            "generated_at": utc_now(),
            "source_id": SOURCE_ID,
            "requested_url": REQUESTED_URL,
            "scope_policy": {
                "full_current_page_by_default": True,
                "section_only_optional": True,
                "replace_links_with_placeholders": True,
                "store_links_separately": True,
                "follow_links": False,
                "download_linked_assets": False,
                "chunk_content": False,
                "infer_edges": False
            },
            "fetch_log": self.fetch_log
        }
        (self.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
        print(json.dumps({"title":meta["title"],"scope":meta["scope"],"links":len(links),"chars":len(text),"output_dir":str(self.output_dir)},indent=2))
        return self.output_dir

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data/nsup")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--pause", type=float, default=0.4)
    ap.add_argument("--section-only", action="store_true")
    a=ap.parse_args()
    try:
        NSUPScraper(Path(a.output_dir),a.timeout,a.pause,a.section_only).run()
        return 0
    except Exception as e:
        print("ERROR:",e,file=sys.stderr); return 1

if __name__=="__main__":
    raise SystemExit(main())
