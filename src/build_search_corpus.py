#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os, re, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urlparse

from bs4 import BeautifulSoup
from openai import OpenAI

PIPELINE_VERSION = "1.0.0"
PROMPT_VERSION = "semantic_chunker_v1"
PLACEHOLDER_RE = re.compile(r"\[\[LINK_\d{4,}\]\]")
TOP_TYPES = {"section", "regulation", "schedule"}

SYSTEM_PROMPT = """You are a semantic segmentation component for a legal and public-procurement retrieval system.
You receive ONE ordered parent segment composed of immutable source blocks.
Your task is ONLY to group contiguous blocks into independently retrievable semantic chunks and generate retrieval metadata.
Do not rewrite, paraphrase, correct, summarize, delete, add, reorder, skip, or duplicate source blocks.
Never alter placeholders matching [[LINK_NNNN]].
Keep legal rules with conditions, exceptions, qualifications, definitions, procedural steps, and consequences where they belong together.
Keep introductory text with the list/table it governs. Respect legal and heading structure.
Prefer semantic coherence over uniform size. Preferred chunk size is roughly 200-800 tokens; coherent chunks up to about 1200 tokens are acceptable.
Return structured JSON only."""

BOUNDARY_SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "start_block_id": {"type": "string"},
                    "end_block_id": {"type": "string"},
                    "retrieval_title": {"type": "string"},
                    "retrieval_summary": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}},
                    "legal_concepts": {"type": "array", "items": {"type": "string"}},
                    "procurement_stage": {"type": "array", "items": {"type": "string"}},
                    "question_intents": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"}
                },
                "required": ["start_block_id","end_block_id","retrieval_title","retrieval_summary","topics","legal_concepts","procurement_stage","question_intents","rationale"],
                "additionalProperties": False
            }
        }
    },
    "required": ["chunks"],
    "additionalProperties": False
}


def now(): return datetime.now(timezone.utc).isoformat()
def clean(s): return re.sub(r"[ \t\r\f\v]+", " ", s or "").strip()
def sha(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()
def safe(s): return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s or "")).strip("_")[:220] or "UNKNOWN"
def phs(s): return sorted(set(PLACEHOLDER_RE.findall(s or "")))
# Placeholder INTEGRITY is measured on raw occurrences, never on the deduplicated set.
# phs() deduplicates, so comparing per-block sets against per-chunk sets reports a false
# PLACEHOLDER_MISMATCH whenever one placeholder occurs in two blocks inside one chunk.
def phs_occ(s): return PLACEHOLDER_RE.findall(s or "")
def strip_ph(s): return clean(PLACEHOLDER_RE.sub("", s or ""))

def read_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def read_jsonl(p):
    p = Path(p)
    if not p.exists(): return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]
def write_jsonl(p, rows):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
def append_jsonl(p, row):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")


def kind(d: Path):
    if (d / "nodes.jsonl").exists(): return "LEGISLATION"
    # Adapter lane: documents normalised from the wider scrape (guidance, policy,
    # workflow, commentary). Same methodology, but link provenance is carried per
    # block instead of as inline [[LINK_NNNN]] placeholders, because that text was
    # extracted without placeholders and must not be mutated after the fact.
    if (d / "blocks.jsonl").exists(): return "BLOCKS"
    if (d / "content_with_link_placeholders.html").exists() or (d / "content_with_link_placeholders.txt").exists(): return "HTML"
    if (d / "pdf_pages.jsonl").exists(): return "PDF"
    return None

def discover(root: Path):
    if kind(root): return [root]
    out=[]; seen=set()
    for p in root.rglob("document.json"):
        d=p.parent
        if kind(d) and str(d.resolve()) not in seen:
            out.append(d); seen.add(str(d.resolve()))
    return sorted(out)

def meta(d: Path):
    m=read_json(d / "document.json") if (d / "document.json").exists() else {}
    m["document_id"] = safe(m.get("document_id") or m.get("source_id") or m.get("source_key") or d.name.upper())
    m.setdefault("title", m.get("name") or d.name)
    m.setdefault("source_url", m.get("root_url") or m.get("canonical_url") or m.get("final_url"))
    return m

def regime(m):
    t=" ".join(str(m.get(k) or "") for k in ["document_id","source_key","title","corpus_role","topic"]).lower()
    if "2015_102" in t or "pcr2015" in t or "public contracts regulations 2015" in t: return "PCR2015"
    if "2024_692" in t or "procurement regulations 2024" in t: return "PR2024"
    if "2023_54" in t or "pa2023" in t or "procurement act 2023" in t: return "PA2023"
    return None

# ---------- Stage 1: parent segments ----------
def child_map(nodes):
    mp=defaultdict(list)
    for n in nodes:
        if n.get("parent_node_id"): mp[str(n["parent_node_id"])].append(n)
    for v in mp.values(): v.sort(key=lambda x:x.get("ordinal",0))
    return mp

def exclusive_text(node, children):
    residual=clean(node.get("text") or "")
    if not children: return residual
    removed=False
    for c in children:
        ct=clean(c.get("text") or "")
        if ct:
            i=residual.find(ct)
            if i>=0:
                residual=clean(residual[:i]+" "+residual[i+len(ct):]); removed=True
    return residual if removed else ""

def legal_blocks(top, cmap):
    blocks=[]; count=0
    def walk(n, ancestors):
        nonlocal count
        ch=cmap.get(str(n["node_id"]),[])
        own=exclusive_text(n,ch)
        if own:
            count+=1
            blocks.append({
                "block_id":f"B{count:04d}","source_node_id":n["node_id"],"block_type":n.get("node_type"),
                "number":n.get("number"),"heading":n.get("heading"),"text":own,
                "legal_path":[{"node_id":a.get("node_id"),"node_type":a.get("node_type"),"number":a.get("number"),"heading":a.get("heading")} for a in ancestors+[n]]
            })
        for c in ch: walk(c,ancestors+[n])
    walk(top,[])
    return blocks

def legal_segments(d,m):
    nodes=read_jsonl(d/"nodes.jsonl"); cmap=child_map(nodes); out=[]
    # Top-level provisions are identified STRUCTURALLY, not by node_type alone.
    # The scrapers label them inconsistently across instruments: PCR2015 carries its
    # 1,120 operative provisions as 'hcontainer' with the numbers attached, while its
    # 'regulation'-typed nodes have no number. Filtering on node_type alone matched
    # only 8 of them, leaving PCR2015 effectively unchunked (26 chunks averaging 2,087
    # tokens). A node id of the form <doc>__<unit>-<number> with no further nesting is
    # a reliable, scraper-independent signal of a top-level provision.
    top_id_re = re.compile(r"__(section|regulation|schedule)-[0-9]+[A-Za-z]?$")
    tops = [
        n for n in nodes
        if n.get("number")
        and "crossheading" not in str(n.get("node_id") or "")
        and (n.get("node_type") in TOP_TYPES or top_id_re.search(str(n.get("node_id") or "")))
    ]
    tops.sort(key=lambda x:x.get("ordinal",0))
    for i,t in enumerate(tops,1):
        blocks=legal_blocks(t,cmap)
        if not blocks: continue
        # Citations are derived from the node ID's own structure, not from node_type.
        # The scrape types schedule paragraphs as "section", so a type-based label gave
        # three different provisions the same citation: PA2023 section 35 (dynamic
        # markets), Schedule 2 paragraph 35 (concession contracts) and Schedule 6
        # paragraph 35 (the national-security mandatory exclusion ground) were all
        # rendered "Procurement Act 2023 s 35". For a system whose value is traceability,
        # an ambiguous citation is a defect: a reader cannot tell which provision they
        # have been given.
        nid = str(t.get("node_id") or "")
        sched = re.search(r"__schedule-([0-9]+[A-Za-z]?)-paragraph-", nid)
        if sched:
            label = f"Sch {sched.group(1)} para"
        elif "__schedule-" in nid and "-paragraph-" not in nid:
            label = "Sch"
        elif re.search(r"__regulation-", nid):
            # PCR2015 carries regulations as 'hcontainer', so a type-derived label
            # produced "hcontainer 9." and "reg SECTION 1". The id is authoritative.
            label = "reg"
        else:
            label={"section":"s","regulation":"reg","schedule":"Sch"}.get(t.get("node_type"),t.get("node_type"))
        # Numbers arrive variously as "9", "9.", "SECTION 1" or "SCHEDULE 6"; normalise to
        # the bare numeral so a citation reads "reg 9", not "reg SECTION 1" or "reg 9.".
        raw_number = str(t.get("number") or "").strip()
        # Keep any alphabetic prefix: Schedule 9 paragraphs A1, B1, C1 and 1 are four
        # different provisions, and stripping to the numeral collapses them onto one.
        m_num = re.search(r"([A-Za-z]?[0-9]+[A-Za-z]?)", raw_number)
        number = m_num.group(1) if m_num else raw_number.rstrip(".")
        out.append({
            "segment_id":f"{safe(m['document_id'])}__SEG_{i:04d}","document_id":m["document_id"],"source_kind":"LEGISLATION",
            "parent_node_id":t["node_id"],"parent_type":t.get("node_type"),"parent_number":t.get("number"),
            "citation":f"{m.get('title')} {label} {number}","heading":t.get("heading"),
            "heading_path":[x for x in [t.get("heading")] if x],"source_url":m.get("root_url") or m.get("source_url"),
            "authority_class":m.get("authority_class"),"corpus_role":m.get("corpus_role"),"legal_regime":regime(m),
            "jurisdiction":m.get("jurisdiction") or "UK","blocks":blocks
        })
    return out

def html_blocks(d):
    hp=d/"content_with_link_placeholders.html"; out=[]
    if hp.exists():
        soup=BeautifulSoup(hp.read_text(encoding="utf-8"),"html.parser")
        for el in soup.find_all(["h1","h2","h3","h4","h5","h6","p","li","tr","blockquote"]):
            txt=clean(el.get_text(" ",strip=True))
            if not txt: continue
            if el.name=="tr":
                cells=[clean(c.get_text(" ",strip=True)) for c in el.find_all(["th","td"],recursive=False)]
                txt=" | ".join([c for c in cells if c]) or txt
            out.append({"block_id":f"B{len(out)+1:04d}","block_type":el.name,"heading_level":int(el.name[1]) if re.fullmatch(r"h[1-6]",el.name) else None,"text":txt})
        return out
    tp=d/"content_with_link_placeholders.txt"
    if tp.exists():
        parts=[clean(x) for x in re.split(r"\n\s*\n",tp.read_text(encoding="utf-8")) if clean(x)]
        for txt in parts:
            mm=re.match(r"^(#{1,6})\s+(.*)$",txt)
            out.append({"block_id":f"B{len(out)+1:04d}","block_type":"heading" if mm else "paragraph","heading_level":len(mm.group(1)) if mm else None,"text":mm.group(2) if mm else txt})
    return out

def group_blocks(blocks,m,source_kind,max_chars):
    groups=[]; cur=[]; chars=0
    for b in blocks:
        major=b.get("block_type") in {"h1","h2","heading"} and (b.get("heading_level") or 2)<=2
        if cur and major and chars>=max_chars//3: groups.append(cur); cur=[]; chars=0
        if cur and chars+len(b.get("text") or "")>max_chars: groups.append(cur); cur=[]; chars=0
        cur.append(b); chars+=len(b.get("text") or "")
    if cur: groups.append(cur)
    out=[]
    for i,g in enumerate(groups,1):
        seg=[]
        for j,b in enumerate(g,1): seg.append({**b,"original_block_id":b["block_id"],"block_id":f"B{j:04d}"})
        heads=[b["text"] for b in g if b.get("block_type") in {"h1","h2","h3","heading"}]
        out.append({
            "segment_id":f"{safe(m['document_id'])}__SEG_{i:04d}","document_id":m["document_id"],"source_kind":source_kind,
            "parent_node_id":None,"parent_type":"DOCUMENT_SECTION" if source_kind=="HTML" else "PDF_PAGE_GROUP",
            "citation":m.get("title"),"heading":heads[0] if heads else m.get("title"),"heading_path":heads[:4],
            "source_url":m.get("canonical_url") or m.get("final_url") or m.get("source_url"),
            "authority_class":m.get("authority_class"),"corpus_role":m.get("evidence_class") or m.get("corpus_role"),
            "legal_regime":regime(m),"jurisdiction":m.get("jurisdiction") or "UK","blocks":seg
        })
    return out

def blocks_segments(d,m,max_chars):
    """Parent segments for adapter-produced documents.

    Groups on heading structure first and size second, so a segment stays a coherent
    unit of guidance rather than an arbitrary character window. Per-block links and
    legal references travel with the block, so a chunk can resolve exactly which
    references it contains.
    """
    blocks=read_jsonl(d/"blocks.jsonl")
    blocks=[b for b in blocks if (b.get("text") or "").strip()]
    if not blocks: return []
    groups=[]; cur=[]; chars=0; cur_head=None
    for b in blocks:
        head=tuple(b.get("heading_path") or [])[:2]
        boundary = cur and head != cur_head and chars >= max_chars//4
        if boundary or (cur and chars+len(b.get("text") or "")>max_chars):
            groups.append(cur); cur=[]; chars=0
        cur.append(b); chars+=len(b.get("text") or ""); cur_head=head
    if cur: groups.append(cur)
    out=[]
    for i,g in enumerate(groups,1):
        seg=[]
        for j,b in enumerate(g,1):
            seg.append({**b,"original_block_id":b.get("block_id"),"block_id":f"B{j:04d}"})
        hp=[x for x in (g[0].get("heading_path") or []) if x][:4]
        out.append({
            "segment_id":f"{safe(m['document_id'])}__SEG_{i:04d}","document_id":m["document_id"],
            "source_kind":"BLOCKS","parent_node_id":None,"parent_type":"DOCUMENT_SECTION",
            "citation":m.get("title"),"heading":(hp[-1] if hp else m.get("title")),"heading_path":hp,
            "source_url":m.get("source_url") or m.get("canonical_url") or m.get("final_url"),
            "authority_class":m.get("authority_class"),"corpus_role":m.get("corpus_role"),
            "legal_regime":m.get("legal_regime"),"jurisdiction":m.get("jurisdiction") or "UK",
            "blocks":seg
        })
    return out

def pdf_segments(d,m,max_chars):
    pages=read_jsonl(d/"pdf_pages.jsonl")
    blocks=[{"block_id":f"P{int(p.get('page_number',i)):04d}","block_type":"pdf_page","page_number":p.get("page_number"),"text":p.get("text") or ""} for i,p in enumerate(pages,1)]
    return group_blocks(blocks,m,"PDF",max_chars)

def stage_segments(root,out,max_chars):
    segs=[]; manifest=[]
    for d in discover(root):
        k=kind(d); m=meta(d)
        ss=(legal_segments(d,m) if k=="LEGISLATION"
            else blocks_segments(d,m,max_chars) if k=="BLOCKS"
            else group_blocks(html_blocks(d),m,"HTML",max_chars) if k=="HTML"
            else pdf_segments(d,m,max_chars))
        for s in ss:
            s["_source_dir"]=str(d.resolve()); s["segment_sha256"]=sha("\n".join(b.get("text") or "" for b in s["blocks"]))
        segs.extend(ss); manifest.append({"source_dir":str(d.resolve()),"document_id":m["document_id"],"source_kind":k,"segment_count":len(ss)})
    out.mkdir(parents=True,exist_ok=True); write_jsonl(out/"parent_segments.jsonl",segs)
    (out/"source_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    return segs

# ---------- Stage 2: LLM boundaries ----------
def validate_plan(seg,plan):
    ids=[b["block_id"] for b in seg["blocks"]]; pos={x:i for i,x in enumerate(ids)}; covered=[]; errs=[]
    if not isinstance(plan.get("chunks"),list) or not plan["chunks"]: return False,["chunks missing or empty"]
    for i,ch in enumerate(plan["chunks"]):
        a,b=ch.get("start_block_id"),ch.get("end_block_id")
        if a not in pos or b not in pos: errs.append(f"chunk {i}: unknown block id"); continue
        if pos[a]>pos[b]: errs.append(f"chunk {i}: start after end"); continue
        covered.extend(ids[pos[a]:pos[b]+1])
    if covered!=ids: errs.append("ranges must cover every source block exactly once and in order")
    return not errs,errs

def llm_payload(seg):
    return json.dumps({"segment_id":seg["segment_id"],"document_id":seg["document_id"],"source_kind":seg["source_kind"],"citation":seg.get("citation"),"heading":seg.get("heading"),"heading_path":seg.get("heading_path",[]),"blocks":[{k:b.get(k) for k in ["block_id","block_type","number","heading","page_number","text"]} for b in seg["blocks"]]},ensure_ascii=False)

def call_llm(client,model,seg,max_output_tokens):
    r=client.responses.create(model=model,instructions=SYSTEM_PROMPT,input=llm_payload(seg),text={"format":{"type":"json_schema","name":"semantic_chunk_boundaries","strict":True,"schema":BOUNDARY_SCHEMA}},max_output_tokens=max_output_tokens,store=False)
    plan=json.loads(r.output_text); usage={}
    if getattr(r,"usage",None):
        for k in ["input_tokens","output_tokens","total_tokens"]:
            v=getattr(r.usage,k,None)
            if v is not None: usage[k]=v
    return plan,{"response_id":getattr(r,"id",None),"model":getattr(r,"model",model),"usage":usage}

def fallback(seg,reason):
    return {"chunks":[{"start_block_id":seg["blocks"][0]["block_id"],"end_block_id":seg["blocks"][-1]["block_id"],"retrieval_title":seg.get("heading") or seg.get("citation") or "Source segment","retrieval_summary":"Source segment retained as one chunk because semantic boundary generation failed validation.","topics":[],"legal_concepts":[],"procurement_stage":[],"question_intents":[],"rationale":f"Fallback: {reason}"}],"_fallback":True}

def deterministic_plan(seg, target_chars):
    """Size-bounded boundary plan produced without the LLM.

    Used for non-prose artefacts - chiefly spreadsheet exports, where one worksheet
    arrives as a single multi-hundred-thousand-character block. Semantic segmentation of
    a data dump buys nothing and cannot fit the model context. Every block is still
    covered exactly once and chunk text is still reconstructed verbatim, so corpus
    invariants hold; only the *metadata* is weaker, which is recorded on each chunk via
    chunking_method so it can be filtered or ablated later.
    """
    chunks=[]; start=None; last=None; acc=0
    for b in seg["blocks"]:
        t=b.get("text") or ""
        if start is None: start=b["block_id"]; acc=0
        acc+=len(t); last=b["block_id"]
        if acc>=target_chars:
            chunks.append({"start_block_id":start,"end_block_id":last,
                "retrieval_title":seg.get("heading") or seg.get("citation") or seg["document_id"],
                "retrieval_summary":"Deterministically split non-prose artefact; no LLM metadata.",
                "topics":[],"legal_concepts":[],"procurement_stage":[],"question_intents":[],
                "rationale":"deterministic size split (oversized/non-prose blocks)"})
            start=None; acc=0
    if start is not None:
        chunks.append({"start_block_id":start,"end_block_id":last,
            "retrieval_title":seg.get("heading") or seg.get("citation") or seg["document_id"],
            "retrieval_summary":"Deterministically split non-prose artefact; no LLM metadata.",
            "topics":[],"legal_concepts":[],"procurement_stage":[],"question_intents":[],
            "rationale":"deterministic size split (oversized/non-prose blocks)"})
    return {"chunks":chunks}


def stage_llm(out,model,max_output_tokens,retries,pause,resume,max_segment_chars=60000,det_target=6000):
    segs=read_jsonl(out/"parent_segments.jsonl"); path=out/"chunk_boundaries.jsonl"
    if not segs: raise RuntimeError("Run segments stage first")
    # Resume must match on CONTENT, not just on segment_id. Segment ids are positional
    # (DOC__SEG_0007), so any change to segmentation renumbers them: a plan computed for
    # the old SEG_0007 would be silently applied to a different provision. Keying on the
    # segment hash makes a stale plan simply miss, and the segment is re-planned.
    existing={}
    if resume and path.exists():
        for x in read_jsonl(path):
            if x.get("segment_id"):
                existing[(x["segment_id"], x.get("segment_sha256"))] = x
    if path.exists() and not resume: path.unlink()
    client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY")); outputs=[]
    for i,seg in enumerate(segs,1):
        key=(seg["segment_id"], seg.get("segment_sha256"))
        if key in existing: outputs.append(existing[key]); print(f"[SKIP] {i}/{len(segs)} {seg['segment_id']}"); continue
        seg_chars=sum(len(b.get("text") or "") for b in seg["blocks"])
        if seg_chars>max_segment_chars:
            plan=deterministic_plan(seg,det_target)
            row={"segment_id":seg["segment_id"],"document_id":seg["document_id"],"segment_sha256":seg["segment_sha256"],
                 "pipeline_version":PIPELINE_VERSION,"prompt_version":"deterministic_size_split_v1","requested_model":None,
                 "created_at":now(),"llm":{"skipped":"segment exceeds max_segment_chars","segment_chars":seg_chars},"plan":plan}
            append_jsonl(path,row); outputs.append(row)
            print(f"[DET] {i}/{len(segs)} {seg['segment_id']} chars={seg_chars} chunks={len(plan['chunks'])}")
            continue
        plan=None; lm={}; err=None
        for a in range(retries+1):
            try:
                cand,lm=call_llm(client,model,seg,max_output_tokens); ok,errs=validate_plan(seg,cand)
                if not ok: raise ValueError("; ".join(errs))
                plan=cand; break
            except Exception as e:
                err=str(e)
                if a<retries: time.sleep(max(1,pause*(a+1)))
        if plan is None: plan=fallback(seg,err or "unknown error"); lm={"error":err}
        row={"segment_id":seg["segment_id"],"document_id":seg["document_id"],"segment_sha256":seg["segment_sha256"],"pipeline_version":PIPELINE_VERSION,"prompt_version":PROMPT_VERSION,"requested_model":model,"created_at":now(),"llm":lm,"plan":plan}
        append_jsonl(path,row); outputs.append(row); print(f"[OK] {i}/{len(segs)} {seg['segment_id']} chunks={len(plan['chunks'])}"); time.sleep(pause)
    return outputs

# ---------- Stage 3: materialize exact chunks ----------
def stage_chunks(out):
    segs=read_jsonl(out/"parent_segments.jsonl"); bmap={x["segment_id"]:x for x in read_jsonl(out/"chunk_boundaries.jsonl")}; chunks=[]; vals=[]
    for seg in segs:
        br=bmap.get(seg["segment_id"])
        if not br: vals.append({"segment_id":seg["segment_id"],"status":"MISSING_BOUNDARIES"}); continue
        if br.get("segment_sha256")!=seg.get("segment_sha256"): vals.append({"segment_id":seg["segment_id"],"status":"INPUT_HASH_MISMATCH"}); continue
        ok,errs=validate_plan(seg,br["plan"])
        if not ok: vals.append({"segment_id":seg["segment_id"],"status":"INVALID_BOUNDARIES","errors":errs}); continue
        ids=[b["block_id"] for b in seg["blocks"]]; pos={x:i for i,x in enumerate(ids)}; source_ph=sorted(p for b in seg["blocks"] for p in phs_occ(b.get("text") or "")); chunk_ph=[]
        for j,bound in enumerate(br["plan"]["chunks"],1):
            selected=seg["blocks"][pos[bound["start_block_id"]]:pos[bound["end_block_id"]]+1]
            text="\n\n".join(b.get("text") or "" for b in selected).strip(); p=phs(text); chunk_ph.extend(phs_occ(text))
            src_nodes=[]
            for b in selected:
                if b.get("source_node_id") and b["source_node_id"] not in src_nodes: src_nodes.append(b["source_node_id"])
            pages=sorted({int(b["page_number"]) for b in selected if b.get("page_number") is not None})
            title=bound.get("retrieval_title") or seg.get("heading") or seg.get("citation"); prefix="\n".join(x for x in [seg.get("citation")," > ".join(seg.get("heading_path") or []),title] if x)
            cid=f"{safe(seg['document_id'])}__{safe(seg['segment_id'])}__CH_{j:03d}"
            chunks.append({"chunk_id":cid,"document_id":seg["document_id"],"segment_id":seg["segment_id"],"chunk_ordinal":j,"parent_node_id":seg.get("parent_node_id"),"parent_type":seg.get("parent_type"),"source_kind":seg.get("source_kind"),"authority_class":seg.get("authority_class"),"corpus_role":seg.get("corpus_role"),"legal_regime":seg.get("legal_regime"),"jurisdiction":seg.get("jurisdiction"),"citation":seg.get("citation"),"heading":seg.get("heading"),"heading_path":seg.get("heading_path",[]),"retrieval_title":bound.get("retrieval_title"),"retrieval_summary":bound.get("retrieval_summary"),"topics":bound.get("topics",[]),"legal_concepts":bound.get("legal_concepts",[]),"procurement_stage":bound.get("procurement_stage",[]),"question_intents":bound.get("question_intents",[]),"start_block_id":bound["start_block_id"],"end_block_id":bound["end_block_id"],"source_node_ids":src_nodes,"page_numbers":pages,"text":text,"embedding_text":clean(prefix+"\n\n"+strip_ph(text)),"link_placeholders":p,"source_url":seg.get("source_url"),"content_sha256":sha(text),"chunking_method":"LLM_SEMANTIC_BOUNDARY_V1","chunking_model":br.get("llm",{}).get("model") or br.get("requested_model"),"prompt_version":br.get("prompt_version"),"pipeline_version":PIPELINE_VERSION})
        vals.append({"segment_id":seg["segment_id"],"status":"PASS" if sorted(chunk_ph)==sorted(source_ph) else "PLACEHOLDER_MISMATCH","source_placeholders":source_ph,"chunk_placeholders":sorted(chunk_ph)})
    write_jsonl(out/"chunks.jsonl",chunks); write_jsonl(out/"chunk_validation.jsonl",vals); return chunks

# ---------- Stage 4: deterministic graph edges ----------
LEG_RE=re.compile(r"https?://(?:www\.)?legislation\.gov\.uk/(?P<type>ukpga|uksi)/(?P<year>\d{4})/(?P<number>\d+)(?:/(?P<kind>section|regulation|article|schedule|part)/(?P<loc>[^/?#]+))?",re.I)
def normloc(x): return re.sub(r"[^A-Za-z0-9]+","",str(x or "")).upper() or None
def docid_url(u):
    m=LEG_RE.search(u or ""); return f"{m.group('type').upper()}_{m.group('year')}_{m.group('number')}" if m else None

def indexes(root):
    dbu={}; prov={}; dirs={}; nodeby={}
    for d in discover(root):
        m=meta(d); did=m["document_id"]; dirs[did]=d
        for k in ["root_url","contents_url","canonical_url","final_url","source_url"]:
            if m.get(k): dbu[urldefrag(str(m[k]))[0].rstrip("/")]=did
        if kind(d)=="LEGISLATION":
            for n in read_jsonl(d/"nodes.jsonl"):
                nodeby[str(n["node_id"])]=n; num=normloc(n.get("number")); nt=str(n.get("node_type") or "").lower()
                if num:
                    keys=[f"{did}|{nt}|{num}"]
                    if nt=="regulation": keys.append(f"{did}|article|{num}")
                    for key in keys: prov[key]=n["node_id"] if key not in prov else None
    return {"doc_by_url":dbu,"prov":prov,"dirs":dirs,"nodeby":nodeby}

def resolve_leg_url(u,idx):
    m=LEG_RE.search(u or "")
    if not m: return {"status":"NOT_LEGISLATION_URL"}
    did=docid_url(u); k=(m.group("kind") or "").lower() or None; loc=normloc(m.group("loc"))
    if did and k and loc:
        keys=[f"{did}|{k}|{loc}"]
        if k=="article": keys.insert(0,f"{did}|regulation|{loc}")
        hits={idx["prov"].get(x) for x in keys if idx["prov"].get(x)}
        if len(hits)==1: return {"status":"EXACT_PROVISION","target_document_id":did,"target_node_id":next(iter(hits))}
        return {"status":"PROVISION_NOT_IN_CORPUS_OR_AMBIGUOUS","target_document_id":did}
    return {"status":"DOCUMENT","target_document_id":did} if did else {"status":"UNRESOLVED"}

def stage_edges(root,out):
    idx=indexes(root); chunks=read_jsonl(out/"chunks.jsonl"); cbd=defaultdict(list)
    for c in chunks: cbd[c["document_id"]].append(c)
    edges=[]; unresolved=[]; keys=set()
    def add(s,r,t,**kw):
        key=(s,r,t)
        if key in keys: return
        keys.add(key); edges.append({"edge_id":f"EDGE_{len(edges)+1:08d}","source_id":s,"relation":r,"target_id":t,**kw})
    for did,d in idx["dirs"].items():
        if kind(d)=="LEGISLATION":
            for n in read_jsonl(d/"nodes.jsonl"):
                add(str(n.get("parent_node_id") or did),"CONTAINS",str(n["node_id"]),evidence_method="PARSED_XML_HIERARCHY",confidence=1.0)
    for c in chunks: add(str(c.get("parent_node_id") or c["document_id"]),"HAS_CHUNK",c["chunk_id"],evidence_method="CHUNK_PROVENANCE",confidence=1.0)
    for did,d in idx["dirs"].items():
        if kind(d)!="LEGISLATION": continue
        for r in read_jsonl(d/"references.jsonl"):
            res=resolve_leg_url(r.get("absolute_url") or r.get("raw_href") or "",idx)
            if res.get("status")=="EXACT_PROVISION": add(r.get("source_node_id"),"CROSS_REFERS_TO",res["target_node_id"],evidence_method=r.get("extraction_method") or "STRUCTURED_XML_REFERENCE",resolution_level="EXACT_PROVISION",confidence=1.0,evidence={"anchor_text":r.get("anchor_text"),"url":r.get("absolute_url")})
            else: unresolved.append({"candidate_type":"STRUCTURED_REFERENCE","source_document_id":did,"source_node_id":r.get("source_node_id"),"raw":r,"resolution":res})
        for r in read_jsonl(d/"regex_reference_candidates.jsonl"):
            km={"SECTION":"section","REGULATION":"regulation","SCHEDULE":"schedule","PART":"part"}; kk=km.get(str(r.get("reference_type") or "").upper()); loc=normloc(r.get("locator")); target=idx["prov"].get(f"{did}|{kk}|{loc}") if kk and loc else None
            if target: add(r.get("source_node_id"),"CROSS_REFERS_TO",target,evidence_method="REGEX_EXACT_INTERNAL_REFERENCE",resolution_level="EXACT_PROVISION",confidence=0.90,evidence={"matched_text":r.get("matched_text"),"locator":r.get("locator")})
            else: unresolved.append({"candidate_type":"REGEX_REFERENCE","source_document_id":did,"source_node_id":r.get("source_node_id"),"raw":r})
    for did,d in idx["dirs"].items():
        if kind(d)!="HTML": continue
        lmap={x.get("placeholder"):x for x in read_jsonl(d/"links.jsonl") if x.get("placeholder")}
        for c in cbd.get(did,[]):
            for p in c.get("link_placeholders",[]):
                lk=lmap.get(p)
                if not lk: unresolved.append({"candidate_type":"PLACEHOLDER_LINK","source_chunk_id":c["chunk_id"],"placeholder":p,"reason":"missing links.jsonl row"}); continue
                u=lk.get("absolute_url") or lk.get("defragmented_url") or ""; res=resolve_leg_url(u,idx)
                if res.get("status")=="EXACT_PROVISION": add(c["chunk_id"],"REFERENCES",res["target_node_id"],evidence_method="HTML_PLACEHOLDER_LINK",resolution_level="EXACT_PROVISION",confidence=1.0,evidence={"placeholder":p,"anchor_text":lk.get("anchor_text"),"url":u,"source_context":lk.get("source_context")}); continue
                td=idx["doc_by_url"].get(urldefrag(u)[0].rstrip("/"))
                if td: add(c["chunk_id"],"REFERENCES",td,evidence_method="HTML_PLACEHOLDER_LINK",resolution_level="DOCUMENT",confidence=1.0,evidence={"placeholder":p,"anchor_text":lk.get("anchor_text"),"url":u,"source_context":lk.get("source_context")})
                else: unresolved.append({"candidate_type":"PLACEHOLDER_LINK","source_chunk_id":c["chunk_id"],"placeholder":p,"url":u,"anchor_text":lk.get("anchor_text"),"source_context":lk.get("source_context"),"resolution":res})
    write_jsonl(out/"edges.jsonl",edges); write_jsonl(out/"unresolved_references.jsonl",unresolved); return edges,unresolved


def manifest(out,root,args):
    files={}
    for n in ["parent_segments.jsonl","chunk_boundaries.jsonl","chunks.jsonl","chunk_validation.jsonl","edges.jsonl","unresolved_references.jsonl"]:
        p=out/n
        if p.exists(): files[n]={"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    (out/"chunking_manifest.json").write_text(json.dumps({"pipeline_version":PIPELINE_VERSION,"prompt_version":PROMPT_VERSION,"generated_at":now(),"input_root":str(root.resolve()),"model":args.model,"max_parent_chars":args.max_parent_chars,"methodology":{"large_parent_segments_first":True,"legislation_parent_unit":"section/regulation/schedule","llm_rewrites_source_text":False,"llm_returns_boundaries_only":True,"exact_text_reconstructed_by_code":True,"link_placeholders_preserved":True,"retrieval_metadata_llm_generated":True,"structural_edges_deterministic":True,"reference_edges_evidence_backed":True,"ambiguous_edges_suppressed":True,"search_unit":"CHUNK","legal_identity_unit":"LEGAL_PROVISION"},"files":files},indent=2,ensure_ascii=False),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=["segments","llm","chunks","edges","all"]); ap.add_argument("--input-root",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--model",default=os.getenv("CHUNK_MODEL")); ap.add_argument("--max-parent-chars",type=int,default=50000); ap.add_argument("--max-output-tokens",type=int,default=8000); ap.add_argument("--retries",type=int,default=2); ap.add_argument("--pause",type=float,default=0.2); ap.add_argument("--resume",action="store_true"); ap.add_argument("--max-segment-chars",type=int,default=60000,help="segments larger than this are split deterministically instead of via the LLM"); ap.add_argument("--deterministic-target-chars",type=int,default=6000); args=ap.parse_args()
    root=Path(args.input_root); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    if args.stage in {"segments","all"}: print(f"[segments] {len(stage_segments(root,out,args.max_parent_chars))}")
    if args.stage in {"llm","all"}:
        if not args.model: raise RuntimeError("--model or CHUNK_MODEL required")
        if not os.environ.get("OPENAI_API_KEY"): raise RuntimeError("OPENAI_API_KEY not set")
        print(f"[llm] {len(stage_llm(out,args.model,args.max_output_tokens,args.retries,args.pause,args.resume,args.max_segment_chars,args.deterministic_target_chars))}")
    if args.stage in {"chunks","all"}: print(f"[chunks] {len(stage_chunks(out))}")
    if args.stage in {"edges","all"}:
        e,u=stage_edges(root,out); print(f"[edges] {len(e)} unresolved={len(u)}")
    manifest(out,root,args); return 0

if __name__=="__main__": raise SystemExit(main())
