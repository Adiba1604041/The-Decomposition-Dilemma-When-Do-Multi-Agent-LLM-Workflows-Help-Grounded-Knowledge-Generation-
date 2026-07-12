import argparse
import json
import re
import time
from typing import Any, Dict, List, Set

import pandas as pd
import requests
from tqdm import tqdm


# -----------------------------
# CS field configuration
# -----------------------------

CS_FIELDS = {
    "Artificial Intelligence": [
        "artificial intelligence",
        "AI",
        "intelligent systems"
    ],
    "Machine Learning": [
        "machine learning",
        "deep learning",
        "neural networks"
    ],
    "Cybersecurity": [
        "cybersecurity",
        "network security",
        "cyber attack"
    ],
    "Data Science": [
        "data science",
        "big data",
        "data analytics"
    ],
    "Human-Computer Interaction": [
        "human computer interaction",
        "HCI",
        "virtual reality",
    ],
}


def clean_text(text: Any) -> str:
    """Normalize whitespace and remove control characters."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def first_value(d: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
    """Return the first available value from possible key names."""
    for key in keys:
        if isinstance(d, dict) and key in d and d[key] not in [None, ""]:
            return d[key]
    return default


def save_records(records: List[Dict[str, Any]], out_prefix: str) -> None:
    """Save records as CSV and JSONL."""
    if not records:
        print("No records found.")
        return

    df = pd.DataFrame(records)

    # Keep only records with abstracts
    df = df[df["abstract"].fillna("").str.len() > 0]

    # Remove global duplicates
    df = df.drop_duplicates(subset=["source", "grant_id"])

    csv_path = f"{out_prefix}.csv"
    jsonl_path = f"{out_prefix}.jsonl"

    df.to_csv(csv_path, index=False)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    print("\nSaved dataset")
    print(f"Total records: {len(df)}")
    print(f"CSV:   {csv_path}")
    print(f"JSONL: {jsonl_path}")

    print("\nRecords per CS field:")
    print(df["cs_field"].value_counts())


# -----------------------------
# NSF extraction
# -----------------------------

def normalize_nsf_award(
    award: Dict[str, Any],
    cs_field: str,
    search_keyword: str
) -> Dict[str, Any]:

    pi_names = first_value(award, ["pdPIName", "pi"], "")

    if isinstance(pi_names, list):
        pi_names = "; ".join(clean_text(p).split("  ")[0] for p in pi_names)

    grant_id = clean_text(first_value(award, ["id"]))

    record = {
        "source": "NSF",
        "cs_field": cs_field,
        "search_keyword": search_keyword,

        "grant_id": grant_id,
        "title": clean_text(first_value(award, ["title"])),
        "abstract": clean_text(first_value(award, ["abstractText"])),

        "pi_names": clean_text(pi_names),
        "institution": clean_text(first_value(award, ["awardeeName", "awardee"])),
        "city": clean_text(first_value(award, ["awardeeCity"])),
        "state": clean_text(first_value(award, ["awardeeStateCode"])),
        "country": clean_text(first_value(award, ["awardeeCountryCode"])),

        "funder": "National Science Foundation",
        "agency": clean_text(first_value(award, ["agency"])),
        "directorate": clean_text(first_value(award, ["orgLongName", "dirAbbr"])),
        "division": clean_text(first_value(award, ["orgLongName2", "divAbbr"])),
        "program": clean_text(first_value(award, ["program", "fundProgramName"])),

        "award_year": clean_text(first_value(award, ["date"]))[-4:],
        "award_date": clean_text(first_value(award, ["date"])),
        "start_date": clean_text(first_value(award, ["startDate"])),
        "end_date": clean_text(first_value(award, ["expDate"])),
        "award_amount": clean_text(
            first_value(award, ["estimatedTotalAmt", "fundsObligatedAmt"])
        ),

        "project_url": f"https://www.nsf.gov/awardsearch/showAward?AWD_ID={grant_id}",
        "raw_json": json.dumps(award, ensure_ascii=False),
    }

    return record


def fetch_nsf_awards_for_keyword(
    keyword: str,
    cs_field: str,
    start_year: int,
    end_year: int,
    target_records: int,
    used_grant_ids: Set[str],
    sleep_seconds: float = 1.0,
    max_pages: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fetch NSF awards for one keyword until target_records are collected
    or max_pages is reached.
    """

    base_url = "https://www.research.gov/awardapi-service/v1/awards.json"

    date_start = f"01/01/{start_year}"
    date_end = f"12/31/{end_year}"

    print_fields = [
        "id",
        "title",
        "abstractText",
        "agency",
        "awardeeName",
        "awardeeCity",
        "awardeeStateCode",
        "awardeeCountryCode",
        "pdPIName",
        "pi",
        "piFirstName",
        "piLastName",
        "date",
        "startDate",
        "expDate",
        "estimatedTotalAmt",
        "fundsObligatedAmt",
        "fundProgramName",
        "program",
        "dirAbbr",
        "divAbbr",
        "orgLongName",
        "orgLongName2",
    ]

    records = []
    offset = 1
    pages_checked = 0

    while len(records) < target_records and pages_checked < max_pages:
        params = {
            "keyword": keyword,
            "dateStart": date_start,
            "dateEnd": date_end,
            "offset": offset,
            "printFields": ",".join(print_fields),
        }

        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        awards = data.get("response", {}).get("award", [])

        if not awards:
            break

        for award in awards:
            grant_id = clean_text(first_value(award, ["id"]))

            if not grant_id:
                continue

            # Avoid duplicates across all fields
            if grant_id in used_grant_ids:
                continue

            record = normalize_nsf_award(
                award=award,
                cs_field=cs_field,
                search_keyword=keyword
            )

            # Keep only records with usable abstracts
            if record["abstract"]:
                records.append(record)
                used_grant_ids.add(grant_id)

            if len(records) >= target_records:
                break

        offset += len(awards)
        pages_checked += 1
        time.sleep(sleep_seconds)

    return records


def fetch_balanced_nsf_dataset(
    records_per_field: int,
    start_year: int,
    end_year: int,
    sleep_seconds: float,
) -> List[Dict[str, Any]]:
    """
    Fetch a balanced NSF dataset:
    20 records per CS field by default.
    """

    all_records = []
    used_grant_ids: Set[str] = set()

    for cs_field, keywords in CS_FIELDS.items():
        print(f"\nCollecting field: {cs_field}")
        field_records = []

        pbar = tqdm(total=records_per_field, desc=cs_field)

        for keyword in keywords:
            remaining = records_per_field - len(field_records)

            if remaining <= 0:
                break

            new_records = fetch_nsf_awards_for_keyword(
                keyword=keyword,
                cs_field=cs_field,
                start_year=start_year,
                end_year=end_year,
                target_records=remaining,
                used_grant_ids=used_grant_ids,
                sleep_seconds=sleep_seconds,
            )

            field_records.extend(new_records)
            pbar.update(len(new_records))

        pbar.close()

        if len(field_records) < records_per_field:
            print(
                f"Warning: only found {len(field_records)} records "
                f"for field '{cs_field}'"
            )

        all_records.extend(field_records[:records_per_field])

    return all_records


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract a balanced NSF GrantDigest dataset."
    )

    parser.add_argument(
        "--records_per_field",
        type=int,
        default=20,
        help="Number of records to collect per CS field."
    )

    parser.add_argument(
        "--start_year",
        type=int,
        default=2021,
        help="NSF start year."
    )

    parser.add_argument(
        "--end_year",
        type=int,
        default=2026,
        help="NSF end year."
    )

    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=1.0,
        help="Delay between API requests."
    )

    parser.add_argument(
        "--out",
        type=str,
        default="grantdigest100_nsf_cs",
        help="Output file prefix. Script writes .csv and .jsonl."
    )

    args = parser.parse_args()

    records = fetch_balanced_nsf_dataset(
        records_per_field=args.records_per_field,
        start_year=args.start_year,
        end_year=args.end_year,
        sleep_seconds=args.sleep_seconds,
    )

    save_records(records, args.out)


if __name__ == "__main__":
    main()