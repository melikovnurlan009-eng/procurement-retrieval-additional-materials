# Source acquisition

The scripts that fetched the corpus documents from their publishers. They are the first stage
of the build described in `corpus/CORPUS_BUILD.md`, feeding the chunkers in `src/`.

| Script | Source family |
|---|---|
| `legislation/group_a_legislation_scraper.py` | legislation.gov.uk — first version |
| `legislation/group_a_legislation_scraper_v2.py` | legislation.gov.uk — second version |
| `../group_a_legislation_scraper_v4.py` | legislation.gov.uk — the version the corpus was built with, kept beside the chunkers it feeds |
| `legislation/scrape_missing_legislation.py` | Instruments the corpus cited but did not hold, ranked by citation frequency |
| `scrape_core_legislation_full.py` | Full-instrument legislation acquisition |
| `scrape_procurement_act_guidance.py` | Procurement Act 2023 guidance on gov.uk |
| `scrape_procurement_policy_notes.py` | Procurement Policy Notes |
| `scrape_procurement_compliance_oversight.py` | Procurement compliance and oversight guidance |
| `scrape_competition_procurement_guidance.py` | Competition and procurement guidance pages |
| `scrape_associated_law_guidance.py` | Associated law: FOIA, data protection, subsidy control |
| `scrape_nsup.py` | National Security Unit for Procurement material |
| `rescrape_procurement_journey.py` | procurementjourney.scot, re-acquired from HTML rather than print renders |
| `scrape_professional_procurement_sources.py` | Practitioner and professional commentary sites |

## What re-running these does and does not give you

These scripts fetch live pages. The published pages have changed since the corpus was built,
so a re-run acquires today's versions of these sources, not the versions the reported results
were computed on. Some of the documents in the frozen corpus are no longer retrievable at
their original URLs at all.

The corpus that the reported results use is therefore identified by the SHA-256 in
`corpus/CORPUS_BUILD.md` rather than by re-acquisition. These scripts are included so the
acquisition is inspectable — which sources were taken, from where, and with what handling —
not as a route to rebuilding the frozen corpus.

Two scraper versions for legislation.gov.uk are included alongside the version used for the
build; the intermediate version between them is not retained. The scrapers for a small number
of source families are likewise not retained, so this directory covers most of the corpus by
document count rather than all of it.

Each script writes normalised JSON per document, which is what the chunkers in `src/` consume.
