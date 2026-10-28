from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _get(url: str, params: dict[str, str]) -> bytes:
    request = Request(
        url + "?" + urlencode(params), headers={"User-Agent": "PedOLGov/0.1 research"}
    )
    with urlopen(request, timeout=60) as response:
        return response.read()


def dailymed_search(drug: str, page_size: int = 20) -> dict:
    return json.loads(
        _get(
            "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json",
            {"drug_name": drug, "pagesize": str(page_size)},
        )
    )


def dailymed_spl(set_id: str) -> bytes:
    from uuid import UUID

    identifier = str(UUID(set_id))
    return _get(f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{identifier}.xml", {})


def pubmed_search(drug: str, indication: str, limit: int = 50) -> list[str]:
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    query = (
        f"({drug}) AND ({indication}) AND "
        "(child OR infant OR adolescent OR pediatrics OR off-label)"
    )
    data = json.loads(
        _get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "sort": "relevance",
                "retmax": str(limit),
            },
        )
    )
    return data["esearchresult"]["idlist"]


def pubmed_abstracts(pmids: list[str]) -> bytes:
    if not pmids or any(not p.isdigit() for p in pmids):
        raise ValueError("PMIDs must be non-empty numeric identifiers")
    return _get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
    )


def openfda_search(search: str, limit: int = 100, skip: int = 0) -> dict:
    if not 1 <= limit <= 100 or skip < 0:
        raise ValueError("invalid pagination")
    return json.loads(
        _get(
            "https://api.fda.gov/drug/event.json",
            {"search": search, "limit": str(limit), "skip": str(skip)},
        )
    )


def dailymed_sections(xml: bytes) -> dict[str, list[str]]:
    import xml.etree.ElementTree as ET

    names = {
        "34067-9": "indications",
        "34070-3": "contraindications",
        "43685-7": "warnings",
        "34081-0": "pediatric_use",
    }
    result = {name: [] for name in names.values()}
    root = ET.fromstring(xml)
    ns = {"hl7": "urn:hl7-org:v3"}
    for section in root.findall(".//hl7:section", ns):
        code = section.find("hl7:code", ns)
        text_node = section.find("hl7:text", ns)
        if code is not None and text_node is not None and code.get("code") in names:
            text = " ".join(" ".join(text_node.itertext()).split())
            if text:
                result[names[code.get("code")]].append(text)
    return result


def parse_pubmed(xml: bytes) -> list[dict]:
    import xml.etree.ElementTree as ET

    articles = []
    root = ET.fromstring(xml)
    for entry in root.findall(".//PubmedArticle"):
        citation = entry.find("MedlineCitation")
        if citation is None:
            continue
        pmid = citation.findtext("PMID", "").strip()
        article = citation.find("Article")
        if not pmid or article is None:
            continue
        abstract = article.find("Abstract")
        if abstract is None:
            continue
        text = " ".join(" ".join(abstract.itertext()).split())
        if not text:
            continue
        types = {
            str(t.text or "").casefold()
            for t in article.findall("./PublicationTypeList/PublicationType")
        }
        descriptors = {
            str(t.text or "").casefold()
            for t in citation.findall("./MeshHeadingList/MeshHeading/DescriptorName")
        }
        if "randomized controlled trial" in types:
            design = "rct"
        elif types & {"observational study", "cohort study", "case-control study"}:
            design = "observational"
        elif "case reports" in types:
            design = "case"
        else:
            design = "unknown"
        pediatric = bool(
            descriptors & {"child", "child, preschool", "infant", "infant, newborn", "adolescent"}
        )
        articles.append({"design": design, "pediatric": pediatric, "pmid": pmid, "text": text})
    return articles
