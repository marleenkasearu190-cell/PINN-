import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.literature_scout import (
    Paper,
    dedupe_papers,
    run_scout,
    score_paper,
)


def test_literature_scout_dedupes_by_doi_and_handles_missing_fields():
    papers = [
        Paper(
            source="openalex",
            source_id="a",
            title="Process-guided lake water temperature modeling",
            year=2024,
            venue="Water Resources Research",
            doi="10.1234/demo",
            abstract="A process-guided model for multi-lake water temperature stratification with open code.",
            citation_count=15,
            is_open_access=True,
        ),
        Paper(
            source="semantic_scholar",
            source_id="b",
            title="Process-guided lake water temperature modeling",
            year=None,
            venue="",
            doi="https://doi.org/10.1234/demo",
            abstract="",
            citation_count=None,
            is_open_access=None,
        ),
        Paper(
            source="crossref",
            source_id="c",
            title="Untitled metadata fallback",
            year=None,
            venue="",
            doi="",
            abstract="",
            citation_count=None,
            is_open_access=None,
        ),
    ]

    deduped = dedupe_papers(papers)

    assert len(deduped) == 2
    merged = next(item for item in deduped if item.doi == "10.1234/demo")
    assert "openalex" in merged.source
    assert "semantic_scholar" in merged.source
    assert merged.citation_count == 15
    assert score_paper(deduped[-1])["overall_score"] >= 0.0


def test_literature_scout_keeps_high_venue_low_relevance_out_of_high_priority():
    paper = Paper(
        source="fixture",
        source_id="nature-unrelated",
        title="Single cell immunotherapy mechanisms in cancer",
        year=2025,
        venue="Nature",
        doi="10.0000/unrelated",
        abstract="A biomedical study with no lake hydrology water temperature or physics transfer evidence.",
        citation_count=500,
        is_open_access=True,
    )

    score = score_paper(paper)

    assert score["venue_quality"] >= 4.0
    assert score["relevance"] < 2.5
    assert score["overall_score"] < 3.5
    assert score["evidence_strength"] == "weak"


def test_literature_scout_flags_fewshot_leakage_and_model_structure_approval():
    paper = Paper(
        source="fixture",
        source_id="fewshot-transformer",
        title="Transformer adapter meta-learning for few-shot lake temperature prediction",
        year=2024,
        venue="Journal of Hydrology",
        doi="10.0000/fewshot",
        abstract=(
            "A few-shot domain adaptation transformer adapter uses future query observations "
            "at test-time for multi-lake water temperature transfer."
        ),
        citation_count=40,
        is_open_access=True,
    )

    score = score_paper(paper)

    assert "support_query_leakage" in score["risk_flags"]
    assert score["recommended_status"] == "needs_approval"


def test_literature_scout_fixture_writes_knowledge_base_only(tmp_path):
    fixture = tmp_path / "papers.json"
    fixture.write_text(
        json.dumps(
            [
                {
                    "source": "fixture",
                    "source_id": "p1",
                    "title": "Physics-informed lake stratification diagnostics",
                    "year": 2023,
                    "venue": "Water Resources Research",
                    "doi": "10.1111/lake",
                    "abstract": (
                        "A physics-informed diagnostic for multi-lake vertical temperature profiles, "
                        "heat budget, stratification, and held-out cross-lake evaluation with open code."
                    ),
                    "citation_count": 25,
                    "is_open_access": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "pipeline" / "literature"

    summary = run_scout(output_root=output_root, fixture=fixture)

    assert summary["deduped_count"] == 1
    assert summary["accepted_count"] == 1
    assert (output_root / "papers.csv").exists()
    assert (output_root / "ideas.csv").exists()
    assert (output_root / "model_hypotheses.md").exists()
    assert list((output_root / "paper_cards").glob("*.md"))
    assert not (tmp_path / "pipeline" / "experiment_queue.csv").exists()
    with (output_root / "ideas.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["recommended_status"] in {"idea_only", "needs_approval"}
    assert "Minimal" not in rows[0]["minimal_experiment"]


def test_literature_scout_rejects_low_relevance_fixture_from_cards(tmp_path):
    fixture = tmp_path / "papers.json"
    fixture.write_text(
        json.dumps(
            [
                {
                    "source": "fixture",
                    "source_id": "good",
                    "title": "Lake water temperature physics-guided stratification model",
                    "year": 2024,
                    "venue": "Journal of Hydrology",
                    "doi": "10.1111/good",
                    "abstract": "A multi-lake water temperature profile model with stratification, heat diagnostics, and heldout evaluation.",
                    "citation_count": 12,
                    "is_open_access": True,
                },
                {
                    "source": "fixture",
                    "source_id": "bad",
                    "title": "Comprehensive integration of single-cell data",
                    "year": 2025,
                    "venue": "Nature",
                    "doi": "10.1111/bad",
                    "abstract": "A biomedical single-cell paper about cellular assays and clinical biomarkers.",
                    "citation_count": 200,
                    "is_open_access": True,
                },
            ]
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "pipeline" / "literature"

    summary = run_scout(output_root=output_root, fixture=fixture, min_relevance=1.5)

    assert summary["deduped_count"] == 2
    assert summary["accepted_count"] == 1
    with (output_root / "papers.csv").open("r", encoding="utf-8", newline="") as handle:
        papers = list(csv.DictReader(handle))
    with (output_root / "rejected.csv").open("r", encoding="utf-8", newline="") as handle:
        rejected = list(csv.DictReader(handle))
    assert papers[0]["source_id"] == "good"
    assert rejected[0]["source_id"] == "bad"
    card_names = [path.name for path in (output_root / "paper_cards").glob("*.md")]
    assert len(card_names) == 1
    assert "single-cell" not in card_names[0]
