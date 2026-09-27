#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag
import requests
from lxml import etree

VERSION='3.0.0'
SOURCES={
 'PA2023':{'document_id':'UKPGA_2023_54','title':'Procurement Act 2023','type':'ukpga','year':'2023','number':'54','root_url':'https://www.legislation.gov.uk/ukpga/2023/54','contents_url':'https://www.legislation.gov.uk/ukpga/2023/54/contents','original_variant':'enacted','expected_type':'section','expected_min':127,'role':'CORE_PRIMARY_LEGISLATION'},
 'PR2024':{'document_id':'UKSI_2024_692','title':'Procurement Regulations 2024','type':'uksi','year':'2024','number':'692','root_url':'https://www.legislation.gov.uk/uksi/2024/692','contents_url':'https://www.legislation.gov.uk/uksi/2024/692/contents/made','original_variant':'made','expected_type':'regulation','expected_min':50,'role':'CORE_SECONDARY_LEGISLATION'},
 'PCR2015':{'document_id':'UKSI_2015_102','title':'Public Contracts Regulations 2015','type':'uksi','year':'2015','number':'102','root_url':'https://www.legislation.gov.uk/uksi/2015/102','contents_url':'https://www.legislation.gov.uk/uksi/2015/102/contents/made','original_variant':'made','expected_type':'regulation','expected_min':120,'role':'LEGACY_PROCUREMENT_LEGISLATION'},
}

def now(): return datetime.now(timezone.utc).isoformat()
def sha(b): return hashlib.sha256(b).hexdigest()
def tsha(s): return hashlib.sha256(s.encode()).hexdigest()
def clean(s): return re.sub(r'\s+',' ',s or '').strip()
def lname(tag):
    if not isinstance(tag,str): return ''
    return tag.rsplit('}',1)[-1]
def attr(el,name):
    for k,v in el.attrib.items():
        if lname(k)==name: return v
    return None
def txt(el): return clean(' '.join(el.itertext()))
def child_text(el,names):
    for c in el.iterchildren():
        if lname(c.tag).lower() in names:
            t=txt(c)
            if t:return t
    return None
def write_jsonl(path,rows):
    with open(path,'w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')

class Scraper:
    def __init__(self,key,out,timeout=90,pause=.35):
        self.key=key; self.s=SOURCES[key]; self.out=out; self.raw=out/'raw'; self.raw.mkdir(parents=True,exist_ok=True)
        self.timeout=timeout; self.pause=pause
        self.session=requests.Session(); self.session.headers.update({'User-Agent':'MastersThesisProcurementResearchBot/3.0','Accept-Language':'en-GB,en;q=0.9'})
    def urls(self):
        r=self.s['root_url'].rstrip('/'); v=self.s['original_variant']
        return [f'{r}/data.akn',f'{r}/data.xml',f'{r}/{v}/data.akn',f'{r}/{v}/data.xml']
    def ntype(self,el):
        n=lname(el.tag).lower()
        m={'part':'part','chapter':'chapter','section':'section','article':'article','subsection':'subsection','paragraph':'paragraph','subparagraph':'subparagraph','point':'point','subpoint':'subpoint','rule':'rule','schedule':'schedule','hcontainer':'hcontainer','p1':'section_or_regulation','p2':'subsection_or_paragraph','p3':'paragraph_or_subparagraph','p4':'subparagraph_or_point'}
        return m.get(n)
    def number(self,el):
        n=child_text(el,{'num','number','pnumber'})
        if n:return n
        for d in el.iterdescendants():
            if lname(d.tag).lower() in {'pnumber','number'}:
                t=txt(d)
                if t:return t
        return None
    def heading(self,el): return child_text(el,{'heading','subheading','title','rubric'})
    def is_top(self,el):
        t=self.ntype(el)
        return t==self.s['expected_type'] or t=='section_or_regulation'
    def score(self,root):
        top=struct=num=0
        for el in root.iter():
            t=self.ntype(el)
            if t: struct+=1
            if t and self.number(el): num+=1
            if self.is_top(el) and self.number(el): top+=1
        tl=len(clean(' '.join(root.itertext())))
        return top*100000+num*100+struct*10+min(tl,10000000),top,tl
    def choose(self):
        report=[]
        for i,u in enumerate(self.urls(),1):
            rec={'url':u}
            try:
                r=self.session.get(u,headers={'Accept':'application/akn+xml, application/xml, text/xml, */*;q=.2'},timeout=self.timeout,allow_redirects=True)
                rec.update(status_code=r.status_code,content_type=r.headers.get('content-type',''),byte_count=len(r.content))
                if r.status_code!=200:
                    rec.update(parse_ok=False,error=f'HTTP {r.status_code}',score=0,top_level_count=0,text_length=0,saved_filename=None); report.append(rec); continue
                fn=f'candidate_{i:02d}.xml'; (self.raw/fn).write_bytes(r.content)
                root=etree.fromstring(r.content,parser=etree.XMLParser(recover=True,huge_tree=True))
                sc,tc,tl=self.score(root)
                rec.update(parse_ok=True,error=None,score=sc,top_level_count=tc,text_length=tl,saved_filename=fn,sha256=sha(r.content))
            except Exception as e:
                rec.update(parse_ok=False,error=str(e),score=0,top_level_count=0,text_length=0,saved_filename=None)
            report.append(rec); time.sleep(self.pause)
        (self.raw/'candidate_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        good=[x for x in report if x.get('parse_ok') and x.get('saved_filename')]
        if not good: raise RuntimeError('No parseable full-body XML candidate. See raw/candidate_report.json')
        good.sort(key=lambda x:(x['score'],x['top_level_count'],x['text_length'],x['byte_count']),reverse=True)
        best=good[0]
        if best['top_level_count']<self.s['expected_min']:
            raise RuntimeError(f"Completeness check failed: {best['top_level_count']} top-level provisions; expected at least {self.s['expected_min']}. See raw/candidate_report.json")
        shutil.copy2(self.raw/best['saved_filename'],self.raw/'selected_source.xml')
        return best
    def structural_children(self,el):
        for c in el:
            if not isinstance(c.tag,str): continue
            if self.ntype(c): yield c
            else: yield from self.structural_children(c)
    def extract(self,root):
        roots=[]
        for el in root.iter():
            if not self.ntype(el): continue
            p=el.getparent(); found=False
            while p is not None:
                if self.ntype(p): found=True; break
                p=p.getparent()
            if not found: roots.append(el)
        nodes=[]; refs=[]; seen=set(); ordinal=0
        def walk(el,parent,depth,path):
            nonlocal ordinal
            if id(el) in seen:return
            seen.add(id(el)); t=self.ntype(el)
            if not t:return
            if t=='section_or_regulation': t='section' if self.s['type']=='ukpga' else 'regulation'
            elif t=='subsection_or_paragraph': t='subsection' if self.s['type']=='ukpga' else 'paragraph'
            elif t=='paragraph_or_subparagraph': t='paragraph'
            elif t=='subparagraph_or_point': t='subparagraph'
            ordinal+=1; num=self.number(el); eid=attr(el,'eId') or attr(el,'id') or attr(el,'Id')
            suffix=re.sub(r'[^A-Za-z0-9_.-]+','_',eid).strip('_') if eid else '_'.join(map(str,path))
            node_id=f"{self.s['document_id']}__{suffix}"
            text=txt(el)
            nodes.append({'node_id':node_id,'document_id':self.s['document_id'],'source_key':self.key,'node_type':t,'number':num,'heading':self.heading(el),'eid':eid,'parent_node_id':parent,'depth':depth,'ordinal':ordinal,'index_path':path,'text':text,'text_sha256':tsha(text),'source_url':self.s['root_url'],'corpus_role':self.s['role'],'chunking_status':'NOT_CHUNKED'})
            ri=0
            for d in el.iter():
                href=attr(d,'href') or attr(d,'Href') or attr(d,'ref')
                if not href: continue
                ri+=1; absolute=urljoin(self.s['root_url'],href); defrag,frag=urldefrag(absolute)
                refs.append({'reference_id':f'{node_id}__REF_{ri:04d}','source_document_id':self.s['document_id'],'source_node_id':node_id,'source_eid':eid,'reference_ordinal':ri,'anchor_text':txt(d),'raw_href':href,'absolute_url':absolute,'defragmented_url':defrag,'fragment':frag or None,'element_name':lname(d.tag),'resolution_status':'UNRESOLVED','target_document_id':None,'target_node_id':None,'edge_status':'NOT_CREATED','extraction_method':'STRUCTURED_XML_REFERENCE'})
            ci=0
            for c in self.structural_children(el):
                p=c.getparent(); between=False
                while p is not None and p is not el:
                    if self.ntype(p): between=True; break
                    p=p.getparent()
                if between: continue
                ci+=1; walk(c,node_id,depth+1,path+[ci])
        for i,r in enumerate(roots,1): walk(r,None,0,[i])
        return nodes,refs
    def run(self):
        best=self.choose(); data=(self.raw/'selected_source.xml').read_bytes(); root=etree.fromstring(data,parser=etree.XMLParser(recover=True,huge_tree=True))
        nodes,refs=self.extract(root); write_jsonl(self.out/'nodes.jsonl',nodes); write_jsonl(self.out/'references.jsonl',refs)
        full=[]
        for n in nodes:
            if n['node_type'] in {'part','chapter','section','regulation','schedule'}:
                label=n['node_type'].upper();
                if n.get('number'): label+=' '+n['number']
                if n.get('heading'): label+=' - '+n['heading']
                full.extend([label,n['text'],''])
        full_text='\n'.join(full); (self.out/'full_text.txt').write_text(full_text,encoding='utf-8')
        counts={}
        for n in nodes: counts[n['node_type']]=counts.get(n['node_type'],0)+1
        top=sum(1 for n in nodes if n['node_type']==self.s['expected_type'])
        summary={'source_key':self.key,'document_id':self.s['document_id'],'title':self.s['title'],'selected_source_url':best['url'],'selected_source_sha256':sha(data),'top_level_type':self.s['expected_type'],'top_level_count':top,'expected_min_top_level':self.s['expected_min'],'completeness_check_passed':top>=self.s['expected_min'],'node_count':len(nodes),'reference_count':len(refs),'node_type_counts':counts,'full_text_char_count':len(full_text)}
        (self.out/'extraction_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        doc={'document_id':self.s['document_id'],'source_key':self.key,'title':self.s['title'],'root_url':self.s['root_url'],'contents_url':self.s['contents_url'],'corpus_role':self.s['role'],'ingestion_scope':'FULL_LEGACY' if self.key=='PCR2015' else 'FULL','authority_class':'PRIMARY_LEGISLATION' if self.s['type']=='ukpga' else 'SECONDARY_LEGISLATION','retrieval_priority':'HIGH_IF_LEGACY' if self.key=='PCR2015' else 'HIGHEST','selected_representation_url':best['url'],'selected_representation_sha256':sha(data),'scraper_version':VERSION,'scraped_at':now(),'chunking_status':'NOT_CHUNKED','graph_edges_status':'NOT_INFERRED'}
        (self.out/'document.json').write_text(json.dumps(doc,indent=2),encoding='utf-8')
        return doc,summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',choices=['ALL','PA2023','PR2024','PCR2015'],default='ALL'); ap.add_argument('--output-dir',default='data/core_legislation'); ap.add_argument('--timeout',type=int,default=90); ap.add_argument('--pause',type=float,default=.35); args=ap.parse_args()
    root=Path(args.output_dir); root.mkdir(parents=True,exist_ok=True); selected=['PA2023','PR2024','PCR2015'] if args.source=='ALL' else [args.source]
    docs=[]; sums=[]; fails=[]
    for key in selected:
        out=root/key.lower(); out.mkdir(parents=True,exist_ok=True)
        try:
            d,s=Scraper(key,out,args.timeout,args.pause).run(); docs.append(d); sums.append(s); print(f"[OK] {key}: {s['top_level_count']} {s['top_level_type']}s, {s['node_count']} structural nodes")
        except Exception as e:
            fails.append({'source_key':key,'url':SOURCES[key]['root_url'],'error':str(e)}); print(f'[ERROR] {key}: {e}',file=sys.stderr)
    write_jsonl(root/'documents.jsonl',docs); write_jsonl(root/'extraction_summaries.jsonl',sums); write_jsonl(root/'failures.jsonl',fails)
    manifest={'scraper_version':VERSION,'generated_at':now(),'requested_sources':selected,'successful_sources':len(docs),'failure_count':len(fails),'methodology':{'official_source':'legislation.gov.uk','full_body_xml_candidates':True,'contents_endpoint_used_as_body':False,'candidate_selection_by_provision_coverage':True,'raw_selected_representation_preserved':True,'structured_hierarchy_extracted':True,'structured_references_preserved':True,'recursive_external_crawl':False,'llm_chunking':False,'semantic_edge_inference':False},'sources':SOURCES}
    (root/'corpus_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); return 2 if fails else 0
if __name__=='__main__': raise SystemExit(main())
