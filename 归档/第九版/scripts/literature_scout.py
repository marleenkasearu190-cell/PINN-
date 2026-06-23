from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import textwrap
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PINN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_ROOT = PINN_ROOT / "pipeline"
DEFAULT_OUTPUT_ROOT = DEFAULT_PIPELINE_ROOT / "literature"
CURRENT_YEAR = 2026
DEFAULT_YEAR_FROM = 2018
DEFAULT_PER_QUERY = 8
DEFAULT_EMAIL = "lake-pinn-literature-scout@example.invalid"
QUERY_THEMES = (
    "lake water temperature physics informed neural network",
    "lake thermal stratification machine learning vertical temperature",
    "process guided deep learning hydrology water temperature",
    "remote sensing lake surface water temperature data assimilation",
    "few shot domain adaptation environmental time series hydrology",
    "uncertainty out of distribution physics informed neural networks hydrology",
    "reservoir thermal dynamics machine learning inflow outflow",
)
HIGH_QUALITY_VENUES = (
    "Water Resources Research",
    "Limnology and Oceanography",
    "Journal of Hydrology",
    "Remote Sensing of Environment",
    "Environmental Modelling & Software",
    "Hydrology and Earth System Sciences",
    "Geoscientific Model Development",
    "Water Research",
    "Earth System Science Data",
    "Environmental Research Letters",
    "Nature Water",
    "Nature Communications",
    "Proceedings of the National Academy of Sciences",
    "IEEE Transactions on Geoscience and Remote Sensing",
)
RELEVANCE_TERMS = (
    "lake",
    "reservoir",
    "water temperature",
    "thermal",
    "stratification",
    "mixing",
    "vertical",
    "profile",
    "surface water temperature",
    "lst",
    "lswt",
    "hydrology",
)
DIRECT_DOMAIN_TERMS = (
    "lake",
    "reservoir",
    "limnology",
    "water temperature",
    "surface water temperature",
    "lake surface",
    "lswt",
    "stratification",
    "vertical profile",
)
BACKGROUND_DOMAIN_TERMS = (
    "hydrology",
    "water resources",
    "environmental system",
    "era5",
    "surface energy",
    "data assimilation",
)
BIOMEDICAL_TERMS = (
    "single-cell",
    "single cell",
    "transcriptomic",
    "microbiome",
    "cancer",
    "clinical",
    "immunotherapy",
    "coronavirus",
    "rt-pcr",
)
METHOD_TERMS = (
    "physics-informed",
    "physics informed",
    "process-guided",
    "process guided",
    "data assimilation",
    "state-space",
    "state space",
    "neural ode",
    "domain adaptation",
    "few-shot",
    "few shot",
    "meta-learning",
    "uncertainty",
    "out-of-distribution",
    "ood",
    "physics-guided",
    "physics guided",
)
PHYSICS_TERMS = (
    "heat",
    "energy",
    "density",
    "mixing",
    "diffusivity",
    "light attenuation",
    "stratification",
    "ice",
    "kz",
    "kd",
)
REPRO_TERMS = ("code", "github", "open source", "data available", "open access", "reproducible")
MODEL_STRUCTURE_TERMS = (
    "transformer",
    "graph neural",
    "gnn",
    "neural operator",
    "foundation model",
    "diffusion",
    "adapter",
    "large language",
)
RISK_PATTERNS = {
    "lake_id_memorization": ("lake id", "site id", "station embedding", "site embedding"),
    "lst_surface_overfit": ("strong lst supervision", "surface-only", "surface only", "satellite target"),
    "kz_kd_bypass": ("unconstrained diffusivity", "free optical", "unbounded mixing", "learned diffusivity"),
    "residual_bypass": ("large residual", "black-box residual", "unconstrained residual", "pure residual"),
    "support_query_leakage": ("future support", "query leakage", "future observations", "transductive test"),
}


@dataclass
class Paper:
    source: str
    source_id: str
    title: str
    year: int | None = None
    venue: str = ""
    doi: str = ""
    url: str = ""
    abstract: str = ""
    citation_count: int | None = None
    is_open_access: bool | None = None
    query: str = ""
    external_ids: dict = field(default_factory=dict)


def _request_json(url: str, *, timeout: int = 20) -> dict:
    request = Request(url, headers={"User-Agent": f"LakePINN literature scout; mailto:{DEFAULT_EMAIL}"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_doi(value: object) -> str:
    text = _normalize_text(value).lower()
    text = text.removeprefix("https://doi.org/").removeprefix("http://dx.doi.org/")
    return text


def _normalize_title(value: object) -> str:
    text = _normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _analysis_text(*parts: object) -> str:
    text = " ".join(_normalize_text(part) for part in parts).lower()
    return re.sub(r"[\u2010-\u2015]", "-", text)


def _slug(value: str, fallback: str = "paper") -> str:
    text = _normalize_title(value).replace(" ", "-")
    return (text[:80].strip("-") or fallback)


def _abstract_from_openalex(record: dict) -> str:
    inverted = record.get("abstract_inverted_index") or {}
    if not isinstance(inverted, dict):
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for pos in positions or []:
            words.append((int(pos), str(word)))
    return " ".join(word for _, word in sorted(words))


def openalex_search(query: str, *, per_query: int, year_from: int, year_to: int) -> list[Paper]:
    params = {
        "search": query,
        "filter": f"from_publication_date:{year_from}-01-01,to_publication_date:{year_to}-12-31",
        "sort": "cited_by_count:desc",
        "per-page": str(per_query),
        "mailto": DEFAULT_EMAIL,
    }
    payload = _request_json(f"https://api.openalex.org/works?{urlencode(params)}")
    papers: list[Paper] = []
    for item in payload.get("results", []):
        venue = (
            item.get("primary_location", {})
            .get("source", {})
            .get("display_name", "")
        )
        doi = _normalize_doi(item.get("doi"))
        papers.append(
            Paper(
                source="openalex",
                source_id=str(item.get("id", "")),
                title=_normalize_text(item.get("title")),
                year=item.get("publication_year"),
                venue=_normalize_text(venue),
                doi=doi,
                url=_normalize_text(item.get("doi") or item.get("id")),
                abstract=_abstract_from_openalex(item),
                citation_count=item.get("cited_by_count"),
                is_open_access=bool((item.get("open_access") or {}).get("is_oa")),
                query=query,
            )
        )
    return papers


def semantic_scholar_search(query: str, *, per_query: int, year_from: int, year_to: int) -> list[Paper]:
    fields = ",".join(
        [
            "title",
            "year",
            "venue",
            "citationCount",
            "abstract",
            "url",
            "externalIds",
            "isOpenAccess",
            "openAccessPdf",
        ]
    )
    params = {
        "query": query,
        "limit": str(per_query),
        "year": f"{year_from}-{year_to}",
        "fields": fields,
    }
    payload = _request_json(f"https://api.semanticscholar.org/graph/v1/paper/search?{urlencode(params)}")
    papers: list[Paper] = []
    for item in payload.get("data", []):
        external_ids = item.get("externalIds") or {}
        doi = _normalize_doi(external_ids.get("DOI"))
        papers.append(
            Paper(
                source="semantic_scholar",
                source_id=str(item.get("paperId", "")),
                title=_normalize_text(item.get("title")),
                year=item.get("year"),
                venue=_normalize_text(item.get("venue")),
                doi=doi,
                url=_normalize_text(item.get("url") or (item.get("openAccessPdf") or {}).get("url")),
                abstract=_normalize_text(item.get("abstract")),
                citation_count=item.get("citationCount"),
                is_open_access=bool(item.get("isOpenAccess")),
                query=query,
                external_ids=external_ids,
            )
        )
    return papers


def crossref_search(query: str, *, per_query: int, year_from: int, year_to: int) -> list[Paper]:
    params = {
        "query.bibliographic": query,
        "rows": str(per_query),
        "filter": f"from-pub-date:{year_from}-01-01,until-pub-date:{year_to}-12-31",
        "sort": "is-referenced-by-count",
        "order": "desc",
        "mailto": DEFAULT_EMAIL,
    }
    payload = _request_json(f"https://api.crossref.org/works?{urlencode(params)}")
    papers: list[Paper] = []
    for item in (payload.get("message") or {}).get("items", []):
        year = None
        parts = ((item.get("published-print") or item.get("published-online") or item.get("issued") or {}).get("date-parts") or [])
        if parts and parts[0]:
            year = parts[0][0]
        titles = item.get("title") or []
        journals = item.get("container-title") or []
        papers.append(
            Paper(
                source="crossref",
                source_id=str(item.get("DOI", "")),
                title=_normalize_text(titles[0] if titles else ""),
                year=year,
                venue=_normalize_text(journals[0] if journals else ""),
                doi=_normalize_doi(item.get("DOI")),
                url=_normalize_text(item.get("URL")),
                abstract=_normalize_text(re.sub("<[^>]+>", " ", item.get("abstract", ""))),
                citation_count=item.get("is-referenced-by-count"),
                is_open_access=bool(item.get("license")),
                query=query,
            )
        )
    return papers


def load_fixture(path: Path) -> list[Paper]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    papers = []
    for item in payload:
        papers.append(
            Paper(
                source=str(item.get("source", "fixture")),
                source_id=str(item.get("source_id", item.get("id", ""))),
                title=_normalize_text(item.get("title")),
                year=item.get("year"),
                venue=_normalize_text(item.get("venue")),
                doi=_normalize_doi(item.get("doi")),
                url=_normalize_text(item.get("url")),
                abstract=_normalize_text(item.get("abstract")),
                citation_count=item.get("citation_count"),
                is_open_access=item.get("is_open_access"),
                query=_normalize_text(item.get("query")),
                external_ids=item.get("external_ids") or {},
            )
        )
    return papers


def collect_papers(
    *,
    queries: Iterable[str],
    per_query: int,
    year_from: int,
    year_to: int,
    sources: Iterable[str],
    sleep_seconds: float = 0.2,
) -> tuple[list[Paper], list[str]]:
    papers: list[Paper] = []
    errors: list[str] = []
    source_set = set(sources)
    for query in queries:
        for source_name, func in (
            ("openalex", openalex_search),
            ("semantic_scholar", semantic_scholar_search),
            ("crossref", crossref_search),
        ):
            if source_name not in source_set:
                continue
            try:
                papers.extend(func(query, per_query=per_query, year_from=year_from, year_to=year_to))
                time.sleep(sleep_seconds)
            except Exception as exc:  # noqa: BLE001 - metadata scout should degrade gracefully.
                errors.append(f"{source_name}:{query}: {exc}")
    return papers, errors


def dedupe_papers(papers: Iterable[Paper]) -> list[Paper]:
    by_key: dict[str, Paper] = {}
    for paper in papers:
        if not paper.title:
            continue
        paper.doi = _normalize_doi(paper.doi)
        key = f"doi:{paper.doi}" if paper.doi else f"title:{_normalize_title(paper.title)}"
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = paper
            continue
        existing.source = ",".join(_unique([*existing.source.split(","), paper.source]))
        existing.source_id = ",".join(_unique([existing.source_id, paper.source_id]))
        existing.venue = existing.venue or paper.venue
        existing.url = existing.url or paper.url
        existing.abstract = existing.abstract or paper.abstract
        existing.query = existing.query or paper.query
        existing.is_open_access = existing.is_open_access if existing.is_open_access is not None else paper.is_open_access
        if paper.citation_count is not None:
            existing.citation_count = max(existing.citation_count or 0, paper.citation_count)
        if paper.year and not existing.year:
            existing.year = paper.year
    return sorted(by_key.values(), key=lambda item: score_paper(item)["overall_score"], reverse=True)


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _normalize_text(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _count_terms(text: str, terms: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def _bounded_score(value: float, *, maximum: float = 5.0) -> float:
    return round(max(0.0, min(maximum, value)), 2)


def _venue_score(venue: str) -> tuple[float, str]:
    if not venue:
        return 1.0, "unknown_venue"
    lowered = venue.lower()
    for good in HIGH_QUALITY_VENUES:
        if good.lower() in lowered:
            return 5.0, "venue_whitelist"
    if any(term in lowered for term in ("nature", "science", "pnas")):
        return 4.0, "broad_high_profile_venue"
    return 2.0, "venue_not_whitelisted"


def detect_risks(text: str) -> list[str]:
    lowered = text.lower()
    risks: list[str] = []
    for risk, patterns in RISK_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            risks.append(risk)
    if "meta-learning" in lowered or "few-shot" in lowered or "few shot" in lowered:
        if any(term in lowered for term in ("test-time", "transductive", "future", "query")):
            risks.append("support_query_leakage")
    return _unique(risks)


def evidence_strength(paper: Paper, text: str, scores: dict[str, float]) -> str:
    lowered = text.lower()
    transfer_evidence = any(term in lowered for term in ("held-out", "heldout", "cross-lake", "regional", "multi-lake", "multi lake"))
    code_or_data = any(term in lowered for term in REPRO_TERMS) or bool(paper.is_open_access)
    if scores["relevance"] >= 4.0 and transfer_evidence and code_or_data:
        return "strong"
    if scores["relevance"] >= 3.0 and scores["method_transferability"] >= 2.0:
        return "medium"
    return "weak"


def score_paper(paper: Paper, *, current_year: int = CURRENT_YEAR) -> dict[str, object]:
    text = _analysis_text(paper.title, paper.abstract, paper.venue)
    relevance = _bounded_score(_count_terms(text, RELEVANCE_TERMS) * 0.8 + _count_terms(text, METHOD_TERMS) * 0.35)
    venue, venue_reason = _venue_score(paper.venue)
    age = max(1, current_year - int(paper.year or current_year) + 1)
    citations_per_year = float(paper.citation_count or 0) / age
    citation_score = _bounded_score(math.log1p(citations_per_year) * 1.25)
    recency = _bounded_score(5.0 - max(0, current_year - int(paper.year or current_year)) * 0.45)
    method_transferability = _bounded_score(_count_terms(text, METHOD_TERMS) * 0.9)
    data_similarity = _bounded_score(_count_terms(text, RELEVANCE_TERMS) * 0.7)
    physical_plausibility = _bounded_score(_count_terms(text, PHYSICS_TERMS) * 0.9)
    reproducibility = _bounded_score(
        _count_terms(text, REPRO_TERMS) * 0.9
        + (1.0 if paper.is_open_access else 0.0)
        + (0.5 if paper.doi else 0.0)
    )
    actionability = _bounded_score(
        method_transferability * 0.45
        + physical_plausibility * 0.35
        + (1.0 if any(term in text.lower() for term in ("ablation", "diagnostic", "loss", "benchmark")) else 0.0)
    )
    risks = detect_risks(text)
    risk_penalty = min(2.0, len(risks) * 0.5)
    overall = _bounded_score(
        relevance * 0.25
        + venue * 0.10
        + citation_score * 0.10
        + recency * 0.10
        + method_transferability * 0.15
        + data_similarity * 0.10
        + physical_plausibility * 0.10
        + reproducibility * 0.05
        + actionability * 0.15
        - risk_penalty
    )
    raw_scores = {
        "relevance": relevance,
        "venue_quality": venue,
        "citation_age_normalized": citation_score,
        "recency": recency,
        "method_transferability": method_transferability,
        "data_similarity": data_similarity,
        "physical_plausibility": physical_plausibility,
        "reproducibility": reproducibility,
        "actionability": actionability,
    }
    status = "needs_approval" if any(term in text.lower() for term in MODEL_STRUCTURE_TERMS) else "idea_only"
    return {
        **raw_scores,
        "overall_score": overall,
        "venue_score_reason": venue_reason,
        "risk_flags": risks,
        "evidence_strength": evidence_strength(paper, text, raw_scores),
        "recommended_status": status,
        "citations_per_year": round(citations_per_year, 2),
    }


def acceptance_reason(paper: Paper, score: dict[str, object], *, min_relevance: float) -> tuple[bool, str]:
    text = _analysis_text(paper.title, paper.abstract, paper.venue)
    direct_domain = any(term in text for term in DIRECT_DOMAIN_TERMS)
    remote_sensing_water = "remote sensing" in text and any(
        term in text for term in ("lake", "reservoir", "water temperature", "surface water temperature", "lswt", "lst")
    )
    background_domain = (any(term in text for term in BACKGROUND_DOMAIN_TERMS) or remote_sensing_water) and (
        any(term in text for term in METHOD_TERMS) or any(term in text for term in PHYSICS_TERMS)
    )
    biomedical = any(term in text for term in BIOMEDICAL_TERMS)
    if biomedical and not direct_domain:
        return False, "biomedical_or_unrelated_domain"
    if float(score["relevance"]) < min_relevance:
        return False, f"relevance_below_{min_relevance}"
    if not (direct_domain or background_domain):
        return False, "missing_lake_pinn_domain_gate"
    if background_domain and not direct_domain and float(score["actionability"]) < 1.0:
        return False, "background_without_actionable_lake_pinn_mechanism"
    return True, "accepted"


def paper_id(paper: Paper) -> str:
    base = paper.doi or _normalize_title(paper.title)
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    return f"{_slug(paper.title)}-{digest}"


def lake_pinn_fit(paper: Paper, score: dict[str, object]) -> str:
    if score["relevance"] >= 4 and score["physical_plausibility"] >= 2:
        return "High: directly relevant to Lake-PINN physics/RECON transfer, still requires ablation."
    if score["method_transferability"] >= 3:
        return "Medium: method may transfer, but data/split assumptions must be checked."
    return "Low: use only as background unless a concrete Lake-PINN diagnostic emerges."


def borrowable_mechanisms(paper: Paper) -> list[str]:
    text = _analysis_text(paper.title, paper.abstract)
    mechanisms = []
    if any(term in text for term in ("loss", "regularization", "physics-informed", "physics informed", "process-guided")):
        mechanisms.append("loss_or_regularization_design")
    if any(term in text for term in ("held-out", "cross-lake", "benchmark", "split")):
        mechanisms.append("split_or_benchmark_design")
    if any(term in text for term in ("uncertainty", "ood", "out-of-distribution")):
        mechanisms.append("uncertainty_or_ood_diagnostic")
    if any(term in text for term in ("mixing", "diffusivity", "light attenuation", "stratification", "heat")):
        mechanisms.append("physical_parameter_constraint")
    if any(term in text for term in ("few-shot", "few shot", "domain adaptation", "meta-learning")):
        mechanisms.append("fewshot_or_domain_adaptation_ablation")
    if any(term in text for term in ("remote sensing", "surface water temperature", "lst", "lswt")):
        mechanisms.append("lst_quality_or_weak_supervision_design")
    if any(term in text for term in ("reservoir", "inflow", "outflow", "water level")):
        mechanisms.append("reservoir_hydrology_diagnostic")
    return mechanisms or ["background_context_only"]


def non_borrow_reasons(paper: Paper, score: dict[str, object]) -> list[str]:
    reasons = []
    if score["relevance"] < 3:
        reasons.append("Low direct relevance to vertical lake temperature RECON or sparse profile transfer.")
    if score["data_similarity"] < 2:
        reasons.append("Data type may not match Lake-PINN profile/LST/ERA5 inputs.")
    if score["physical_plausibility"] < 2:
        reasons.append("Physical constraints are unclear; avoid using it to justify Kz/Kd/residual freedom.")
    if score["reproducibility"] < 2:
        reasons.append("Reproducibility signal is weak from available metadata.")
    for risk in score["risk_flags"]:
        reasons.append(f"Risk flag: {risk}.")
    return reasons or ["No immediate blocker, but still require a small transfer-valid ablation before adoption."]


def minimal_experiment(paper: Paper, score: dict[str, object]) -> str:
    mechanisms = set(borrowable_mechanisms(paper))
    if "physical_parameter_constraint" in mechanisms:
        return "Run an L4 LOCAL34 ablation with fixed split: compare constrained Kz/Kd or heat-loss diagnostic against current clean-physics baseline."
    if "fewshot_or_domain_adaptation_ablation" in mechanisms:
        return "Run an L7 support-count ablation on 0/1/3/5 profiles with strict support-before-query separation."
    if "lst_quality_or_weak_supervision_design" in mechanisms:
        return "Run an L4 LST robustness ablation: observed-only, quality flags, dropout 0.2/0.4, and weak surface loss."
    if "uncertainty_or_ood_diagnostic" in mechanisms:
        return "Run an L8 diagnostic-only uncertainty/OOD smoke on existing checkpoints; do not retrain first."
    if "reservoir_hydrology_diagnostic" in mechanisms:
        return "Run a reservoir subset diagnostic first; queue hydrology data work as needs_approval before model changes."
    if "loss_or_regularization_design" in mechanisms:
        return "Run an L3/L4 clean-physics ablation on LOCAL34: add only the proposed loss/regularizer, keep split and checkpoint selection fixed."
    return "Write as background hypothesis only; require one L1/L3 smoke-sized ablation before any structure change."


def idea_row(paper: Paper, score: dict[str, object]) -> dict[str, str]:
    mechanisms = borrowable_mechanisms(paper)
    l_stage = "L4"
    if any("fewshot" in item for item in mechanisms):
        l_stage = "L7"
    elif any("uncertainty" in item for item in mechanisms):
        l_stage = "L8"
    elif any("split" in item for item in mechanisms):
        l_stage = "L5"
    elif any("reservoir" in item for item in mechanisms):
        l_stage = "L0"
    return {
        "paper_id": paper_id(paper),
        "title": paper.title,
        "stage": l_stage,
        "evidence_strength": str(score["evidence_strength"]),
        "overall_score": str(score["overall_score"]),
        "borrowable_mechanisms": ";".join(mechanisms),
        "risk_flags": ";".join(score["risk_flags"]),
        "recommended_status": str(score["recommended_status"]),
        "minimal_experiment": minimal_experiment(paper, score),
    }


def paper_to_row(paper: Paper, score: dict[str, object]) -> dict[str, str]:
    return {
        "paper_id": paper_id(paper),
        "title": paper.title,
        "year": "" if paper.year is None else str(paper.year),
        "venue": paper.venue,
        "doi": paper.doi,
        "url": paper.url,
        "source": paper.source,
        "source_id": paper.source_id,
        "query": paper.query,
        "citation_count": "" if paper.citation_count is None else str(paper.citation_count),
        "citations_per_year": str(score["citations_per_year"]),
        "is_open_access": "" if paper.is_open_access is None else str(bool(paper.is_open_access)).lower(),
        "overall_score": str(score["overall_score"]),
        "relevance": str(score["relevance"]),
        "venue_quality": str(score["venue_quality"]),
        "citation_age_normalized": str(score["citation_age_normalized"]),
        "method_transferability": str(score["method_transferability"]),
        "data_similarity": str(score["data_similarity"]),
        "physical_plausibility": str(score["physical_plausibility"]),
        "reproducibility": str(score["reproducibility"]),
        "actionability": str(score["actionability"]),
        "evidence_strength": str(score["evidence_strength"]),
        "risk_flags": ";".join(score["risk_flags"]),
        "recommended_status": str(score["recommended_status"]),
        "impact_factor_note": "not_fetched_do_not_infer_if_without_jcr_or_user_supplied_table",
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(rows[0].keys())
    else:
        fields = []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_card(path: Path, paper: Paper, score: dict[str, object]) -> None:
    reasons = non_borrow_reasons(paper, score)
    mechanisms = borrowable_mechanisms(paper)
    abstract = paper.abstract or "No abstract available from open metadata."
    content = f"""# {paper.title}

## Metadata
- Year: {paper.year or "unknown"}
- Venue: {paper.venue or "unknown"}
- DOI: {paper.doi or "missing"}
- URL: {paper.url or "missing"}
- Sources: {paper.source}
- Citation count: {paper.citation_count if paper.citation_count is not None else "missing"}
- Open access metadata: {paper.is_open_access}
- Impact factor note: not fetched; do not infer IF without JCR or a user-supplied table.

## Claim
{_summarize_claim(paper)}

## Evidence And Split
{_summarize_evidence(paper)}

## Lake-PINN Fit
{lake_pinn_fit(paper, score)}

## Borrowable Mechanisms
{_bullet_list(mechanisms)}

## Do Not Borrow Directly
{_bullet_list(reasons)}

## Minimal Validation Experiment
{minimal_experiment(paper, score)}

## Critical Scores
- Overall: {score["overall_score"]}
- Evidence strength: {score["evidence_strength"]}
- Recommended status: {score["recommended_status"]}
- Risk flags: {"; ".join(score["risk_flags"]) or "none"}

## Abstract
{textwrap.fill(abstract, width=100)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _bullet_list(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _summarize_claim(paper: Paper) -> str:
    text = paper.abstract or paper.title
    first = re.split(r"(?<=[.!?])\s+", text.strip())[0]
    return first[:500] if first else "No claim extractable from metadata; inspect the paper before using it."


def _summarize_evidence(paper: Paper) -> str:
    text = " ".join([paper.title, paper.abstract]).lower()
    notes = []
    if "multi" in text and "lake" in text:
        notes.append("Metadata suggests multi-lake evidence.")
    if any(term in text for term in ("held-out", "heldout", "cross-lake", "domain")):
        notes.append("Metadata suggests some transfer or heldout evaluation.")
    if any(term in text for term in ("profile", "vertical", "stratification")):
        notes.append("Metadata touches vertical/profile structure.")
    if not notes:
        notes.append("No clear split or dataset evidence from open metadata; treat as weak until inspected.")
    return " ".join(notes)


def write_hypotheses(path: Path, ideas: list[dict[str, str]]) -> None:
    lines = [
        "# Lake-PINN Literature-Derived Hypotheses",
        "",
        "These are candidate hypotheses only. They must not trigger model changes or formal training without approval.",
        "",
    ]
    for row in ideas[:20]:
        lines.extend(
            [
                f"## {row['paper_id']}",
                f"- Title: {row['title']}",
                f"- Stage: {row['stage']}",
                f"- Evidence: {row['evidence_strength']}",
                f"- Score: {row['overall_score']}",
                f"- Status: {row['recommended_status']}",
                f"- Risks: {row['risk_flags'] or 'none'}",
                f"- Minimal experiment: {row['minimal_experiment']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    papers: list[Paper],
    output_root: Path,
    *,
    min_relevance: float = 1.5,
    prune_stale_cards: bool = True,
) -> dict[str, str]:
    scored_all = [(paper, score_paper(paper)) for paper in papers]
    scored_all.sort(key=lambda item: item[1]["overall_score"], reverse=True)
    accepted_pairs: list[tuple[Paper, dict[str, object], str]] = []
    rejected_pairs: list[tuple[Paper, dict[str, object], str]] = []
    for paper, score in scored_all:
        accepted, reason = acceptance_reason(paper, score, min_relevance=min_relevance)
        if accepted:
            accepted_pairs.append((paper, score, reason))
        else:
            rejected_pairs.append((paper, score, reason))
    scored = [(paper, score) for paper, score, _ in accepted_pairs]
    rejected = [
        {
            **paper_to_row(paper, score),
            "rejection_reason": reason,
        }
        for paper, score, reason in rejected_pairs
    ]
    paper_rows = [paper_to_row(paper, score) for paper, score in scored]
    idea_rows = [idea_row(paper, score) for paper, score in scored]
    write_csv(output_root / "papers.csv", paper_rows)
    write_csv(output_root / "ideas.csv", idea_rows)
    write_csv(output_root / "rejected.csv", rejected)
    cards_dir = output_root / "paper_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    if prune_stale_cards:
        current_names = {f"{paper_id(paper)}.md" for paper, _ in scored}
        for old_card in cards_dir.glob("*.md"):
            if old_card.name not in current_names:
                old_card.unlink()
    for paper, score in scored:
        write_card(cards_dir / f"{paper_id(paper)}.md", paper, score)
    write_hypotheses(output_root / "model_hypotheses.md", idea_rows)
    return {
        "papers_csv": str((output_root / "papers.csv").resolve()),
        "ideas_csv": str((output_root / "ideas.csv").resolve()),
        "rejected_csv": str((output_root / "rejected.csv").resolve()),
        "paper_cards_dir": str(cards_dir.resolve()),
        "model_hypotheses": str((output_root / "model_hypotheses.md").resolve()),
    }


def run_scout(
    *,
    output_root: Path,
    queries: Iterable[str] = QUERY_THEMES,
    per_query: int = DEFAULT_PER_QUERY,
    year_from: int = DEFAULT_YEAR_FROM,
    year_to: int = CURRENT_YEAR,
    sources: Iterable[str] = ("openalex", "semantic_scholar", "crossref"),
    fixture: Path | None = None,
    max_papers: int = 50,
    min_relevance: float = 1.5,
    prune_stale_cards: bool = True,
) -> dict:
    errors: list[str] = []
    if fixture:
        raw_papers = load_fixture(fixture)
    else:
        raw_papers, errors = collect_papers(
            queries=queries,
            per_query=per_query,
            year_from=year_from,
            year_to=year_to,
            sources=sources,
        )
    papers = dedupe_papers(raw_papers)[:max_papers]
    outputs = write_outputs(
        papers,
        output_root,
        min_relevance=min_relevance,
        prune_stale_cards=prune_stale_cards,
    )
    return {
        "date": date.today().isoformat(),
        "output_root": str(output_root.resolve()),
        "raw_count": len(raw_papers),
        "deduped_count": len(papers),
        "accepted_count": len(list(csv.DictReader((output_root / "papers.csv").open(encoding="utf-8")))) if (output_root / "papers.csv").exists() else 0,
        "errors": errors,
        "outputs": outputs,
        "note": "Literature scout writes ideas only; it does not modify models, splits, manifests, or training queues.",
    }


def _parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect and critically score Lake-PINN literature ideas.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument("--queries", default="")
    parser.add_argument("--sources", default="openalex,semantic_scholar,crossref")
    parser.add_argument("--per-query", type=int, default=DEFAULT_PER_QUERY)
    parser.add_argument("--max-papers", type=int, default=50)
    parser.add_argument("--min-relevance", type=float, default=1.5)
    parser.add_argument("--year-from", type=int, default=DEFAULT_YEAR_FROM)
    parser.add_argument("--year-to", type=int, default=CURRENT_YEAR)
    parser.add_argument("--dry-run", action="store_true", help="Write literature artifacts only; never alter models/training.")
    parser.add_argument("--keep-stale-cards", action="store_true")
    args = parser.parse_args(argv)
    queries = _parse_csv_list(args.queries) if args.queries else QUERY_THEMES
    summary = run_scout(
        output_root=args.output_root,
        queries=queries,
        per_query=args.per_query,
        year_from=args.year_from,
        year_to=args.year_to,
        sources=_parse_csv_list(args.sources),
        fixture=args.fixture,
        max_papers=args.max_papers,
        min_relevance=args.min_relevance,
        prune_stale_cards=not args.keep_stale_cards,
    )
    summary["dry_run"] = bool(args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
