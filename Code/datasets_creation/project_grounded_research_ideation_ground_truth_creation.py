import argparse
import json
import os
import time
import pandas as pd
from tqdm import tqdm
import openai


client = openai.OpenAI(api_key="your_key_here")  


# -----------------------------
# JSON schema
# -----------------------------

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_type": {
            "type": "string",
            "enum": [
                "GrantFuture: Research Opportunity Generation from Funded Project Records"
            ]
        },
        "source_record": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "grant_id": {"type": "string"},
                "cs_field": {"type": "string"},
                "title": {"type": "string"},
                "abstract": {"type": "string"},
                "pi_names": {"type": "string"},
                "institution": {"type": "string"},
                "funder": {"type": "string"},
                "program": {"type": "string"},
                "award_year": {"type": "string"}
            },
            "required": [
                "grant_id",
                "cs_field",
                "title",
                "abstract",
                "pi_names",
                "institution",
                "funder",
                "program",
                "award_year"
            ]
        },
        "source_anchors": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "original_research_area": {"type": "string"},
                "original_problem": {"type": "string"},
                "original_approach": {"type": "string"},
                "original_expected_contribution": {"type": "string"}
            },
            "required": [
                "original_research_area",
                "original_problem",
                "original_approach",
                "original_expected_contribution"
            ]
        },
        "future_direction_requirements": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "must_connect_to_original_project": {
                    "type": "boolean",
                    "enum": [True]
                },
                "must_identify_a_research_gap_or_extension": {
                    "type": "boolean",
                    "enum": [True]
                },
                "must_propose_a_new_future_direction": {
                    "type": "boolean",
                    "enum": [True]
                },
                "must_include_potential_objectives": {
                    "type": "boolean",
                    "enum": [True]
                },
                "must_include_possible_methodology": {
                    "type": "boolean",
                    "enum": [True]
                },
                "must_explain_expected_contribution": {
                    "type": "boolean",
                    "enum": [True]
                },
                "must_explain_broader_impact": {
                    "type": "boolean",
                    "enum": [True]
                }
            },
            "required": [
                "must_connect_to_original_project",
                "must_identify_a_research_gap_or_extension",
                "must_propose_a_new_future_direction",
                "must_include_potential_objectives",
                "must_include_possible_methodology",
                "must_explain_expected_contribution",
                "must_explain_broader_impact"
            ]
        },
        "acceptable_direction_types": {
            "type": "array",
            "items": {"type": "string"}
        },
        "forbidden_outputs": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "task_type",
        "source_record",
        "source_anchors",
        "future_direction_requirements",
        "acceptable_direction_types",
        "forbidden_outputs"
    ]
}


# -----------------------------
# Prompt templates
# -----------------------------

DEVELOPER_PROMPT = """
You are an expert research evaluation assistant.

Your task is to create a rubric-based ground truth reference for a dataset called GrantFuture.

GrantFuture asks a system to generate a future research direction inspired by a funded NSF project record.

Your goal is NOT to write the future research direction.
Your goal is to create a structured evaluation reference that a judge can use to evaluate generated future directions.

The evaluation reference must capture:
- the original research area
- the original problem
- the original approach
- the original expected contribution
- the required properties of a valid future direction
- acceptable types of future directions
- forbidden outputs that should be penalized

Important:
- Use only the provided funded project metadata and abstract.
- Do not add outside knowledge.
- Do not invent missing technical details.
- Do not write a proposal.
- Do not write the future direction itself.
- Produce evaluable constraints, not prose narration.
- The output must follow the provided JSON schema exactly.
"""

USER_PROMPT = """
Here is the funded NSF project record:

{SOURCE_RECORD}

Create a ground truth JSON object with this exact conceptual structure:

{{
  "task_type": "GrantFuture: Research Opportunity Generation from Funded Project Records",

  "source_record": {{
    "grant_id": "...",
    "cs_field": "...",
    "title": "...",
    "abstract": "...",
    "pi_names": "...",
    "institution": "...",
    "funder": "...",
    "program": "...",
    "award_year": "..."
  }},

  "source_anchors": {{
    "original_research_area": "...",
    "original_problem": "...",
    "original_approach": "...",
    "original_expected_contribution": "..."
  }},

  "future_direction_requirements": {{
    "must_connect_to_original_project": true,
    "must_identify_a_research_gap_or_extension": true,
    "must_propose_a_new_future_direction": true,
    "must_include_potential_objectives": true,
    "must_include_possible_methodology": true,
    "must_explain_expected_contribution": true,
    "must_explain_broader_impact": true
  }},

  "acceptable_direction_types": [
    "..."
  ],

  "forbidden_outputs": [
    "..."
  ]
}}

Ground truth construction rules:
- The source_record must preserve the original metadata exactly.
- source_anchors must be faithful to the title and abstract.
- original_research_area should identify the main research topic.
- original_problem should identify the main challenge or gap addressed by the funded project.
- original_approach should describe the project method, system, framework, dataset, study, or technical plan.
- original_expected_contribution should describe the expected contribution of the funded project.
- acceptable_direction_types should list reasonable future extensions of this specific project.
- forbidden_outputs should list invalid behaviors, such as summarizing the original project, proposing an unrelated topic, inventing completed results, or claiming the future direction is already funded.
"""


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_source_record(row):
    return {
        "grant_id": clean_value(row.get("grant_id", "")),
        "cs_field": clean_value(row.get("cs_field", "")),
        "title": clean_value(row.get("title", "")),
        "abstract": clean_value(row.get("abstract", "")),
        "pi_names": clean_value(row.get("pi_names", "")),
        "institution": clean_value(row.get("institution", "")),
        "funder": clean_value(row.get("funder", "National Science Foundation")),
        "program": clean_value(row.get("program", "")),
        "award_year": clean_value(row.get("award_year", ""))
    }


def build_user_prompt(source_record):
    return USER_PROMPT.format(
        SOURCE_RECORD=json.dumps(source_record, indent=2, ensure_ascii=False)
    )


def call_gpt5_mini_with_schema(developer_prompt, user_prompt):
    response = client.responses.create(
        model="gpt-5-mini",
        instructions=developer_prompt,
        input=user_prompt,
        store=False,
        text={
            "format": {
                "type": "json_schema",
                "name": "grantfuture_ground_truth",
                "schema": SCHEMA,
                "strict": True
            }
        }
    )

    parsed = json.loads(response.output_text)
    return parsed


def validate_metadata_preserved(gt, source_record):
   
    for key in [
        "grant_id",
        "cs_field",
        "title",
        "abstract",
        "pi_names",
        "institution",
        "funder",
        "program",
        "award_year"
    ]:
        if gt["source_record"].get(key, "") != source_record.get(key, ""):
            raise ValueError(
                f"Metadata mismatch for {key}. "
                f"Expected: {source_record.get(key, '')} | "
                f"Got: {gt['source_record'].get(key, '')}"
            )


def generate_ground_truth_for_row(row, max_retries=3, sleep_seconds=1.0):
    source_record = build_source_record(row)
    user_prompt = build_user_prompt(source_record)

    last_error = ""

    for attempt in range(max_retries + 1):
        try:
            gt = call_gpt5_mini_with_schema(
                developer_prompt=DEVELOPER_PROMPT,
                user_prompt=user_prompt
            )

            validate_metadata_preserved(gt, source_record)

            return gt, "successful"

        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            grant_id = source_record.get("grant_id", "")
            print(
                f"Attempt {attempt}/{max_retries} failed "
                f"for grant_id={grant_id}: {last_error}"
            )
            time.sleep(sleep_seconds)

    return None, last_error


# -----------------------------
# Dataset runner
# -----------------------------

def generate_ground_truth_dataset(
    input_csv,
    output_csv,
    output_jsonl,
    max_retries=3,
    sleep_seconds=1.0,
    limit=None
):
    df = pd.read_csv(input_csv)

    if limit is not None:
        df = df.head(limit).copy()

    ground_truth_json_list = []
    ground_truth_status_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating ground truth"):
        gt, status = generate_ground_truth_for_row(
            row=row,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds
        )

        if gt is None:
            ground_truth_json_list.append("")
            ground_truth_status_list.append(status)
        else:
            ground_truth_json_list.append(json.dumps(gt, ensure_ascii=False))
            ground_truth_status_list.append("successful")

        time.sleep(sleep_seconds)

    df["ground_truth_json"] = ground_truth_json_list
    df["ground_truth_status"] = ground_truth_status_list

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_csv, index=False)

    jsonl_dir = os.path.dirname(output_jsonl)
    if jsonl_dir:
        os.makedirs(jsonl_dir, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    print("\nFinished generating GrantFuture ground truth.")
    print(f"Total rows: {len(df)}")
    print(f"Successful rows: {(df['ground_truth_status'] == 'successful').sum()}")
    print(f"Failed rows: {(df['ground_truth_status'] != 'successful').sum()}")
    print(f"CSV saved to: {output_csv}")
    print(f"JSONL saved to: {output_jsonl}")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate strict-schema GPT-5-mini ground truth for GrantFuture."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the NSF grant dataset CSV."
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save CSV with ground_truth_json."
    )

    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Path to save JSONL output."
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries per row."
    )

    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=1.0,
        help="Delay between API calls."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: run only first N rows for testing."
    )

    args = parser.parse_args()

    generate_ground_truth_dataset(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_jsonl=args.output_jsonl,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        limit=args.limit
    )
    # Final dataset is renamed to "project_grounded_research_ideation_ground_truth_creation.csv" in the github repository