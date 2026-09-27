#!/usr/bin/env python3
"""Per-source content filters: keep the meaningful parts, drop the page furniture.

Scope of what this can fix
--------------------------
LLM labelling of 900 chunks found 21% GOOD, 56% INCOMPLETE, 23% LOW_VALUE. Only the
LOW_VALUE class is addressable here. INCOMPLETE is dominated by "ends mid-sentence" and
appears in every domain including parsed legislation (41%); it is caused by where chunk
boundaries fall, not by what the scraper collected, and filtering cannot repair it.

So this module targets the 23%: navigation, advice footers, contact blocks, copyright
notices, contents listings, biographies and extraction artifacts.

Design
------
Rules are DROP rules over individual blocks, applied before chunking, plus text repairs
applied to surviving blocks. Each rule is either universal or scoped to the domain whose
labelled failures motivated it, so a pattern that is boilerplate on one site cannot silently
remove content on another.

Every rule was derived from labelled examples rather than intuition, and the module is
validated against those labels: `validate()` reports how much LOW_VALUE it catches and,
more importantly, how much GOOD it wrongly drops. A filter that removes substantive material
is worse than no filter, so the false-positive rate is the number that governs the
thresholds here.

Three extraction corruptions are repaired rather than filtered, because the underlying text
is sound:
    word \n space \n word     assets.publishing PDFs, 284 chunks, 1.9M chars
    word \t \r nbsp word      a second assets.publishing pattern
    li duplicated as p        HTML normaliser, 11.6% of list items
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- text repair
RE_NLSP = re.compile(r"(?<=\w)\n \n(?=\w)")
RE_TRNB = re.compile(r"[\t\r ]+")
RE_WORDLINE = re.compile(r"(?<=[a-z,;])\n(?=[a-z])")
RE_MULTISPACE = re.compile(r"[  ]{2,}")
RE_MULTINL = re.compile(r"\n{3,}")


def repair_text(t: str) -> str:
    """Undo layout artifacts. Never changes wording, only whitespace."""
    if not t:
        return t
    t = RE_NLSP.sub(" ", t)
    t = RE_TRNB.sub(" ", t)
    t = RE_WORDLINE.sub(" ", t)
    t = RE_MULTISPACE.sub(" ", t)
    t = RE_MULTINL.sub("\n\n", t)
    return t.strip()


def drop_adjacent_duplicates(blocks: list[dict]) -> list[dict]:
    """The HTML normaliser emits list items twice, as <li> then identical <p>."""
    out = []
    for b in blocks:
        t = (b.get("text") or "").strip()
        if out and t == (out[-1].get("text") or "").strip():
            continue
        out.append(b)
    return out


# ---------------------------------------------------------------- drop rules
UNIVERSAL = [
    ("chrome", re.compile(r"^\s*(open or close|back to top|view pdf|download pdf|print this page|"
                          r"skip to (main )?content|expand all|collapse all|toggle navigation|"
                          r"share this page|menu|search|home)\s*$", re.I)),
    ("copyright", re.compile(r"(©|\(c\))\s*crown copyright|open government licence|"
                             r"nationalarchives\.gov\.uk/doc/open-government-licence", re.I)),
    ("cookie", re.compile(r"\b(cookies? (on|policy|settings)|accept (all )?cookies|"
                          r"we use cookies)\b", re.I)),
    ("contact_block", re.compile(r"^\s*(contact|get in touch|how we can help)\b.{0,80}"
                                 r"(\+?\d[\d\s]{8,}|@[\w.]+\.\w+)", re.I | re.S)),
    ("phone_list", re.compile(r"(\+44|\+?07)\d[\d\s]{7,}.*(\+44|\+?07)\d[\d\s]{7,}", re.S)),
    ("filename_artifact", re.compile(r"\.(docx|pdf|xlsx|pptx)\s*$", re.I)),
]

DOMAIN_RULES = {
    "procurementpathway.civilservice.gov.uk": [
        # 57% LOW_VALUE: advice footers and topic listings repeated on every page.
        ("advice_footer", re.compile(r"additional support and guidance|make sure you:\s*$|"
                                     r"seek legal and commercial advice", re.I)),
        ("topic_listing", re.compile(r"^\s*other guidance relevant to this area\b", re.I)),
        ("applies_to_trailer", re.compile(r"^\s*this applies to\s*$", re.I)),
    ],
    "procurementjourney.scot": [
        # 29% LOW_VALUE: accordion chrome and answers that only defer to a lawyer.
        ("defer_to_lawyer", re.compile(r"^\s*you should seek (specific )?legal advice[^.]*\.\s*$", re.I)),
        ("route_nav", re.compile(r"^\s*route [123]\b.{0,40}$", re.I)),
    ],
    "procurementlawyers.org.uk": [
        # 43% LOW_VALUE: association news, prizes, member biographies.
        ("association_news", re.compile(r"essay prize|memorial (prize|lecture)|annual (dinner|meeting)|"
                                        r"membership (application|renewal)|committee members?\b", re.I)),
        ("biography", re.compile(r"\b(read law at|was called to the bar|is a partner at|"
                                 r"joined the (firm|chambers))\b", re.I)),
    ],
    "procurementportal.com": [
        ("marketing", re.compile(r"^\s*(how we can help you|book a demo|request a callback|"
                                 r"our services)\b", re.I)),
    ],
    "gov.uk": [
        ("withdrawn_notice", re.compile(r"this (note|guidance|publication) is (now )?"
                                        r"(out of date|withdrawn)", re.I)),
    ],
    "assets.publishing.service.gov.uk": [
        ("toc", re.compile(r"^\s*(contents|table of contents)\s*$", re.I)),
        ("doc_header", re.compile(r"^\s*\S+\.(docx|pdf)\s*$", re.I)),
    ],
    "judiciary.uk": [
        ("title_page", re.compile(r"^\s*(title page|front cover)\s*$", re.I)),
    ],
}

# A block with none of these is unlikely to state a rule, a procedure or a definition.
SUBSTANCE = re.compile(r"\b(must|shall|may|means|applies|require|entitled|prohibit|"
                       r"is to be|are to be|should|will be|provides?|sets? out)\b", re.I)
MIN_SUBSTANTIVE_CHARS = 45


def drop_reason(text: str, domain: str = "", block_type: str = "") -> str | None:
    """Return the rule name if this block should be dropped, else None."""
    t = (text or "").strip()
    if not t:
        return "empty"
    for name, rx in UNIVERSAL:
        if rx.search(t):
            return name
    for name, rx in DOMAIN_RULES.get(domain, []):
        if rx.search(t):
            return name
    # Very short blocks with no modal or definitional verb carry no rule. Headings are
    # exempt: they are kept deliberately, to be grouped with the text they govern.
    if len(t) < MIN_SUBSTANTIVE_CHARS and not block_type.startswith("h") and not SUBSTANCE.search(t):
        return "thin_no_substance"
    return None


def filter_blocks(blocks: list[dict], domain: str = "") -> tuple[list[dict], dict]:
    """Repair, deduplicate and filter a document's blocks. Returns (kept, drop_counts)."""
    import collections
    dropped = collections.Counter()
    out = []
    for b in drop_adjacent_duplicates(blocks):
        b = dict(b)
        b["text"] = repair_text(b.get("text") or "")
        why = drop_reason(b["text"], domain, str(b.get("block_type") or ""))
        if why:
            dropped[why] += 1
            continue
        out.append(b)
    return out, dict(dropped)


# ------------------------------------------------------- ingestion routing
import re as _re

STATUTORY_XML = _re.compile(
    r"legislation\.gov\.uk/.*?/data\.(akn|xml)$|legislation\.gov\.uk/[a-z]+/\d{4}/[\w-]+/?$",
    _re.I)


def route(url: str) -> str:
    """Which ingestion lane should this URL take?

    Statutory XML must go to the Akoma Ntoso / CLML parser, never to the general block
    adapter. The adapter flattens the AkN tree and silently drops nested <paragraph>
    content: measured, 270 chunks from legislation.gov.uk entered through it, leaving
    stems such as "The safeguards must consist of or include measures which—" with their
    (a)(b)(c) items absent from the text entirely. The same misrouting produced a second,
    coarser copy of the Procurement Act competing with the parsed one.

    Returns "LEGISLATION" or "BLOCKS".
    """
    return "LEGISLATION" if STATUTORY_XML.search((url or "").strip()) else "BLOCKS"
