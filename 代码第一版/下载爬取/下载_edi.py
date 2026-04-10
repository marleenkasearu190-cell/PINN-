import argparse
import calendar
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd


PASTA_BASE = "https://pasta.lternet.edu"
USER_AGENT = "Mozilla/5.0 (EDI lake downloader)"
PROJECT_DIR = Path(__file__).resolve().parent
TEMPERATURE_COLUMN_CANDIDATES = [
    "Temperature_C",
    "Temperature_degCelsius",
    "Temperature",
    "Temp_C",
    "Temp",
]
DATE_COLUMN_CANDIDATES = ["Date", "date", "datetime", "DateTime", "DATETIME"]


@dataclass
class Entity:
    name: str
    url: str
    md5: str | None = None


@dataclass
class SearchResult:
    package_id: str
    title: str
    keywords: list[str]


@dataclass
class PackageCandidate:
    score: int
    result: SearchResult
    doi: str | None
    entities: list[Entity]


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def download_file(url: str, target: Path) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response, target.open("wb") as fh:
        fh.write(response.read())


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def ask_nonempty_input(prompt_text: str) -> str:
    while True:
        value = input(prompt_text).strip()
        if value:
            return value
        print("输入不能为空，请重新输入。")


def truncate_text(text: str, max_length: int = 110) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def ask_candidate_selection(candidates: list[PackageCandidate], lake_name: str) -> PackageCandidate:
    print(f"\n找到多个与 {lake_name} 相关的 EDI 数据包候选：")
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"  {index}. {candidate.result.package_id} | score={candidate.score} | "
            f"files={len(candidate.entities)} | {truncate_text(candidate.result.title)}"
        )

    print("直接回车将使用推荐候选 1，也可以输入编号手动选择。")
    while True:
        raw = input("请输入候选编号 [默认 1]: ").strip()
        if raw == "":
            return candidates[0]
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(candidates):
                return candidates[choice - 1]
        print(f"请输入 1 到 {len(candidates)} 之间的编号，或直接回车。")


def parse_package_id(package_id: str) -> tuple[str, int, int]:
    parts = package_id.strip().split(".")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid package id: {package_id}. Expected format like 'edi.552.1'."
        )
    scope, identifier, revision = parts
    return scope, int(identifier), int(revision)


def metadata_url(scope: str, identifier: int, revision: int) -> str:
    return f"{PASTA_BASE}/package/metadata/eml/{scope}/{identifier}/{revision}"


def search_url(query: str, rows: int) -> str:
    params = {
        "defType": "edismax",
        "q": query,
        "fl": "packageid,title,keyword",
        "sort": "score,desc",
        "rows": rows,
    }
    return f"{PASTA_BASE}/package/search/eml?{urlencode(params)}"


def detect_date_column(columns: Iterable[str]) -> str:
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(
        "Could not find a date column. Expected one of: "
        + ", ".join(DATE_COLUMN_CANDIDATES)
    )


def detect_long_temperature_column(columns: Iterable[str]) -> str | None:
    for candidate in TEMPERATURE_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def parse_entities(xml_text: str) -> tuple[str, str | None, list[Entity]]:
    root = ET.fromstring(xml_text)
    title = root.findtext(".//{*}dataset/{*}title") or "Unknown title"

    doi = None
    for alt in root.findall(".//{*}alternateIdentifier"):
        system = alt.attrib.get("system", "")
        text = (alt.text or "").strip()
        if "doi.org" in system.lower() and text:
            doi = text.replace("doi:", "https://doi.org/")
            break

    entities: list[Entity] = []
    for table in root.findall(".//{*}dataTable"):
        name = table.findtext("./{*}entityName") or table.findtext("./{*}physical/{*}objectName")
        url = table.findtext(".//{*}distribution/{*}online/{*}url")
        checksum = None
        for auth in table.findall("./{*}physical/{*}authentication"):
            if auth.attrib.get("method", "").upper() == "MD5":
                checksum = (auth.text or "").strip()
                break
        if name and url:
            entities.append(Entity(name=name, url=url, md5=checksum))

    return title, doi, entities


def parse_search_results(xml_text: str) -> list[SearchResult]:
    root = ET.fromstring(xml_text)
    results: list[SearchResult] = []
    for document in root.findall("./document"):
        package_id = (document.findtext("./packageid") or "").strip()
        title = (document.findtext("./title") or "").strip()
        keywords = [
            (item.text or "").strip()
            for item in document.findall("./keywords/keyword")
            if (item.text or "").strip()
        ]
        if package_id:
            results.append(SearchResult(package_id=package_id, title=title, keywords=keywords))
    return results


def search_packages_by_lake_name(lake_name: str, rows: int = 20) -> list[SearchResult]:
    query = lake_name.strip()
    if not query:
        raise ValueError("Lake name cannot be empty.")
    xml_text = fetch_text(search_url(query, rows))
    return parse_search_results(xml_text)


def ensure_downloads(outdir: Path, entities: Iterable[Entity]) -> list[Path]:
    raw_dir = outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for entity in entities:
        target = raw_dir / entity.name
        needs_download = True
        if target.exists() and entity.md5:
            needs_download = md5sum(target).lower() != entity.md5.lower()
        elif target.exists():
            needs_download = False

        if needs_download:
            print(f"Downloading {entity.name} ...")
            download_file(entity.url, target)
        else:
            print(f"Skipping {entity.name}, already exists.")

        if entity.md5:
            actual = md5sum(target)
            if actual.lower() != entity.md5.lower():
                raise ValueError(
                    f"Checksum mismatch for {target.name}: expected {entity.md5}, got {actual}"
                )
        downloaded.append(target)

    return downloaded


def longest_consecutive_streak(dates: pd.DatetimeIndex) -> int:
    if len(dates) == 0:
        return 0
    longest = 1
    current = 1
    previous = dates[0]
    for current_date in dates[1:]:
        if (current_date - previous).days == 1:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
        previous = current_date
    return max(longest, current)


def max_gap_between_observations(dates: pd.DatetimeIndex) -> int:
    if len(dates) <= 1:
        return 0
    diffs = pd.Series(dates).diff().dt.days.dropna()
    return int(diffs.max()) if not diffs.empty else 0


def prepare_temperature_profile_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    date_column = detect_date_column(df.columns)
    data = df.copy()
    data["_parsed_date"] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna(subset=["_parsed_date"]).sort_values("_parsed_date").copy()

    wide_temp_columns = [col for col in data.columns if col.startswith("Temp_")]
    if wide_temp_columns:
        data = data.dropna(subset=wide_temp_columns, how="all").copy()
        return data, date_column

    long_temp_column = detect_long_temperature_column(data.columns)
    if "Depth_m" in data.columns and long_temp_column:
        data["Depth_m"] = pd.to_numeric(data["Depth_m"], errors="coerce")
        data[long_temp_column] = pd.to_numeric(data[long_temp_column], errors="coerce")
        data = data.dropna(subset=["Depth_m", long_temp_column]).copy()
        return data, date_column

    raise ValueError(
        "The selected temperature profile file is not in a supported format. "
        "Expected wide-format Temp_* columns, or long-format Date/Depth_m/Temperature_* columns."
    )


def summarize_temperature_years(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    data, date_column = prepare_temperature_profile_frame(df)
    data["year"] = data["_parsed_date"].dt.year

    rows = []
    for year, group in data.groupby("year"):
        observed_dates = pd.DatetimeIndex(group["_parsed_date"].dt.normalize().unique()).sort_values()
        expected_days = 366 if calendar.isleap(year) else 365
        first_obs = observed_dates[0]
        last_obs = observed_dates[-1]
        start_of_year = pd.Timestamp(year=year, month=1, day=1)
        end_of_year = pd.Timestamp(year=year, month=12, day=31)
        observed_days = len(observed_dates)
        coverage_ratio = observed_days / expected_days
        span_days = (last_obs - first_obs).days + 1
        rows.append(
            {
                "year": year,
                "rows": len(group),
                "observed_days": observed_days,
                "expected_days": expected_days,
                "coverage_ratio": round(coverage_ratio, 6),
                "longest_streak_days": longest_consecutive_streak(observed_dates),
                "max_gap_between_obs_days": max_gap_between_observations(observed_dates),
                "first_obs": first_obs.date().isoformat(),
                "last_obs": last_obs.date().isoformat(),
                "days_missing_at_start": (first_obs - start_of_year).days,
                "days_missing_at_end": (end_of_year - last_obs).days,
                "span_days": span_days,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        [
            "coverage_ratio",
            "longest_streak_days",
            "span_days",
            "max_gap_between_obs_days",
        ],
        ascending=[False, False, False, True],
    )
    return summary, date_column


def pick_best_year(summary: pd.DataFrame) -> int:
    if summary.empty:
        raise ValueError("No yearly summary rows were produced.")
    return int(summary.iloc[0]["year"])


def infer_dataset_tag(
    dataset_name: str | None,
    title: str,
    scope: str,
    identifier: int,
    revision: int,
) -> str:
    raw = dataset_name or title
    tag = sanitize_name(raw)
    if not tag:
        tag = f"{scope}_{identifier}_{revision}"
    return tag[:80].strip("_") or f"{scope}_{identifier}_{revision}"


def save_metadata(
    outdir: Path,
    title: str,
    doi: str | None,
    entities: list[Entity],
    scope: str,
    identifier: int,
    revision: int,
    dataset_tag: str,
) -> Path:
    payload = {
        "package_id": f"{scope}.{identifier}.{revision}",
        "title": title,
        "doi": doi,
        "metadata_url": metadata_url(scope, identifier, revision),
        "entities": [
            {"name": entity.name, "url": entity.url, "md5": entity.md5}
            for entity in entities
        ],
    }
    metadata_path = outdir / f"{dataset_tag}_edi_metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metadata_path


def looks_like_temperature_profile(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    try:
        sample = pd.read_csv(path, nrows=5)
    except Exception:
        return False

    try:
        prepare_temperature_profile_frame(sample)
        return True
    except Exception:
        return False


def entity_name_score(name: str) -> int:
    lowered = name.lower()
    score = 0
    if "temperatureprofiles" in lowered:
        score += 80
    if "temperature" in lowered and "profile" in lowered:
        score += 60
    if "temp_" in lowered:
        score += 25
    if lowered.endswith(".csv"):
        score += 10
    return score


def candidate_score(result: SearchResult, lake_name: str, entities: list[Entity]) -> int:
    lake_tokens = [token for token in re.split(r"[^a-z0-9]+", lake_name.lower()) if token]
    haystacks = [result.title.lower(), " ".join(result.keywords).lower()]
    score = 0

    for token in lake_tokens:
        if any(token in haystack for haystack in haystacks):
            score += 12
    if lake_name.lower() in result.title.lower():
        score += 25
    if "lake" in result.title.lower():
        score += 5

    entity_scores = [entity_name_score(entity.name) for entity in entities]
    if entity_scores:
        score += max(entity_scores)
        score += min(len(entity_scores), 10)
    return score


def build_candidates(lake_name: str, search_rows: int = 20) -> list[PackageCandidate]:
    results = search_packages_by_lake_name(lake_name, rows=search_rows)
    if not results:
        raise ValueError(f"No EDI packages were found for lake name: {lake_name}")

    candidates: list[PackageCandidate] = []
    seen_package_ids: set[str] = set()
    for result in results:
        if result.package_id in seen_package_ids:
            continue
        seen_package_ids.add(result.package_id)

        try:
            scope, identifier, revision = parse_package_id(result.package_id)
            xml_text = fetch_text(metadata_url(scope, identifier, revision))
            title, doi, entities = parse_entities(xml_text)
        except Exception:
            continue

        resolved_result = SearchResult(
            package_id=result.package_id,
            title=title or result.title,
            keywords=result.keywords,
        )
        candidates.append(
            PackageCandidate(
                score=candidate_score(resolved_result, lake_name, entities),
                result=resolved_result,
                doi=doi,
                entities=entities,
            )
        )

    candidates.sort(key=lambda item: (-item.score, item.result.package_id))
    return candidates


def resolve_package_from_search(
    lake_name: str,
    search_rows: int = 20,
) -> tuple[str, str, str | None, list[Entity]]:
    candidates = build_candidates(lake_name, search_rows=search_rows)
    if not candidates:
        raise ValueError(
            f"Search found packages for {lake_name}, but none of them could be opened successfully."
        )

    if len(candidates) == 1:
        chosen = candidates[0]
        print(f"只找到 1 个候选数据包，自动使用: {chosen.result.package_id}")
    else:
        shown_candidates = candidates[:8]
        chosen = ask_candidate_selection(shown_candidates, lake_name)
        print(f"已选择数据包: {chosen.result.package_id}")

    return chosen.result.package_id, chosen.result.title or lake_name, chosen.doi, chosen.entities


def select_temperature_file(
    downloaded_files: list[Path],
    preferred_pattern: str | None = None,
) -> Path:
    csv_files = [path for path in downloaded_files if path.suffix.lower() == ".csv"]
    if not csv_files:
        raise ValueError("No CSV files were downloaded from the package.")

    if preferred_pattern:
        matched = [
            path for path in csv_files if preferred_pattern.lower() in path.name.lower()
        ]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            names = ", ".join(path.name for path in matched)
            raise ValueError(
                f"Multiple CSV files matched --temperature-pattern '{preferred_pattern}': {names}"
            )

    for path in csv_files:
        if "temperatureprofiles" in path.name.lower():
            return path

    matched_profiles = [path for path in csv_files if looks_like_temperature_profile(path)]
    if len(matched_profiles) == 1:
        return matched_profiles[0]
    if len(matched_profiles) > 1:
        names = ", ".join(path.name for path in matched_profiles)
        raise ValueError(
            "Found multiple temperature-profile-like CSV files. "
            f"Please rerun with --temperature-pattern. Candidates: {names}"
        )

    raise ValueError(
        "Could not identify a temperature profile file in the downloaded data. "
        "Please rerun with --temperature-pattern to specify the correct CSV."
    )


def build_default_outdir(dataset_tag: str) -> Path:
    return PROJECT_DIR / "data" / dataset_tag


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download an EDI lake dataset and find the year closest to a full continuous year."
        )
    )
    parser.add_argument(
        "--package-id",
        default=None,
        help="EDI package id, for example 'edi.552.1'. If omitted, the script searches by lake name.",
    )
    parser.add_argument(
        "--lake-name",
        default=None,
        help="Lake name used to search EDI packages when --package-id is not provided.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Short lake or dataset name used in output file names. Defaults to the lake name.",
    )
    parser.add_argument(
        "--temperature-pattern",
        default=None,
        help="Optional substring used to pick the correct temperature profile CSV.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory for downloaded files and outputs. Defaults to data/<dataset-name>.",
    )
    parser.add_argument(
        "--search-rows",
        type=int,
        default=20,
        help="Number of EDI search results to inspect when auto-searching by lake name.",
    )
    args = parser.parse_args()

    lake_name = args.lake_name
    if not args.package_id and not lake_name:
        lake_name = ask_nonempty_input("请输入要下载的湖泊名称: ")

    if args.package_id:
        scope, identifier, revision = parse_package_id(args.package_id)
        requested_metadata_url = metadata_url(scope, identifier, revision)
        print(f"Fetching metadata from {requested_metadata_url} ...")
        xml_text = fetch_text(requested_metadata_url)
        title, doi, entities = parse_entities(xml_text)
        resolved_package_id = args.package_id
    else:
        print(f"Searching EDI packages for lake name: {lake_name} ...")
        resolved_package_id, title, doi, entities = resolve_package_from_search(
            lake_name,
            search_rows=args.search_rows,
        )
        scope, identifier, revision = parse_package_id(resolved_package_id)
        print(f"Resolved package id: {resolved_package_id}")

    if not entities:
        raise ValueError("No downloadable data tables were found in the EDI metadata.")

    dataset_name = args.dataset_name or lake_name or title
    dataset_tag = infer_dataset_tag(dataset_name, title, scope, identifier, revision)
    outdir = Path(args.outdir).resolve() if args.outdir else build_default_outdir(dataset_tag)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {title}")
    print(f"Package: {scope}.{identifier}.{revision}")
    if doi:
        print(f"DOI: {doi}")
    print(f"Output directory: {outdir}")
    print("Entities found:")
    for entity in entities:
        print(f"  - {entity.name}")

    metadata_path = save_metadata(
        outdir,
        title,
        doi,
        entities,
        scope,
        identifier,
        revision,
        dataset_tag,
    )
    downloaded_files = ensure_downloads(outdir, entities)

    temperature_file = select_temperature_file(downloaded_files, args.temperature_pattern)
    print(f"Analyzing yearly coverage in {temperature_file.name} ...")

    temperature_df = pd.read_csv(temperature_file)
    summary, date_column = summarize_temperature_years(temperature_df)
    best_year = pick_best_year(summary)

    processed_dir = outdir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary_path = processed_dir / f"{dataset_tag}_temperature_year_summary.csv"
    summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)

    best_year_df = temperature_df.copy()
    best_year_df["_parsed_date"] = pd.to_datetime(best_year_df[date_column], errors="coerce")
    best_year_df = best_year_df[best_year_df["_parsed_date"].dt.year == best_year].sort_values("_parsed_date")
    best_year_df = best_year_df.drop(columns=["_parsed_date"])
    best_year_path = processed_dir / f"{dataset_tag}_temperature_profiles_best_year_{best_year}.csv"
    best_year_df.to_csv(best_year_path, index=False, quoting=csv.QUOTE_MINIMAL)

    best_row = summary.iloc[0]
    print("\nBest year candidate:")
    print(
        f"  {best_year} | coverage={best_row['coverage_ratio']:.3f} | "
        f"observed_days={int(best_row['observed_days'])}/{int(best_row['expected_days'])} | "
        f"longest_streak={int(best_row['longest_streak_days'])} | "
        f"max_gap={int(best_row['max_gap_between_obs_days'])}"
    )
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved yearly summary to: {summary_path}")
    print(f"Saved best-year subset to: {best_year_path}")


if __name__ == "__main__":
    main()
