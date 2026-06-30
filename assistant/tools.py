# assistant/tools.py
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "exports"


def _load_jobs() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "jobs.csv")


def _load_job_skills() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "job_skills.csv")

def _apply_skill_filter(df: pd.DataFrame, skill: str) -> pd.DataFrame:
    if skill is None:
        return df
    job_skills_df = _load_job_skills()
    merged = pd.merge(df, job_skills_df, on='job_id')
    return merged[merged["skill"].str.contains(skill, case=False, na=False)]


def get_top_skills(
    role: str = None,
    region: str = None,
    workplace_type: str = None,
    top_n: int = 10
) -> str:
    jobs_df = _load_jobs()
    jobs_skills_df = _load_job_skills()

    merged = pd.merge(jobs_df, jobs_skills_df, on='job_id')
    if role is not None:
        merged = merged[merged["job_title"].str.contains(role, case=False, na=False)]
    if region is not None:
        merged = merged[merged["region"].str.contains(region, case=False, na=False)]
    if workplace_type is not None:
        merged = merged[merged["workplace_type"].str.contains(workplace_type, case=False, na=False)]
    
    result = (
        merged.groupby('skill')
        .size()
        .reset_index(name='count')
        .sort_values(by='count', ascending=False)
        .head(top_n)
    )

    if result.empty:
        return "No skills found for the given filters."
    else:
        return result.to_string(index=False)


def get_top_employers(
    skill: str = None,
    region: str = None,
    top_n: int = 10
) -> str:
    jobs_df = _load_jobs()
    merged = _apply_skill_filter(jobs_df, skill)

    if region is not None:
        merged = merged[merged["region"].str.contains(region, case=False, na=False)]

    result = (
        merged.groupby('employer')
        .size()
        .reset_index(name='count')
        .sort_values(by='count', ascending=False)
        .head(top_n)
    )

    if result.empty:
        return "No employers found for the given filters."
    else:
        return result.to_string(index=False)
   

def get_top_locations(
    skill: str = None,
    workplace_type: str = None,
    top_n: int = 10
) -> str:
    jobs_df = _load_jobs()
    merged = _apply_skill_filter(jobs_df, skill)

    if workplace_type is not None:
        merged = merged[merged["workplace_type"].str.contains(workplace_type, case=False, na=False)]
    
    result = (
        merged.groupby(["region", "municipality"])
        .size()
        .reset_index(name='count')
        .sort_values(by='count', ascending=False)
        .head(top_n)
    )

    if result.empty:
        return "No locations found for the given filters."
    else:
        return result.to_string(index=False)


def get_workplace_type_distribution(
    region: str = None,
    skill: str = None
) -> str:

    jobs_df = _load_jobs()
    if region is not None:
        jobs_df = jobs_df[jobs_df["region"].str.contains(region, case=False, na=False)]
    merged = _apply_skill_filter(jobs_df, skill)
    
    result = (
        merged.groupby("workplace_type")
        .size()
        .reset_index(name='count')
        .sort_values(by='count', ascending=False)
    )

    if result.empty:
        return "No workplace type distribution found for the given filters."
    else:
        return result.to_string(index=False)


def get_experience_distribution(
    role: str = None,
    region: str = None
) -> str:
    jobs_df = _load_jobs()

    if role is not None:
        jobs_df = jobs_df[jobs_df["job_title"].str.contains(role, case=False, na=False)]
    if region is not None:
        jobs_df = jobs_df[jobs_df["region"].str.contains(region, case=False, na=False)]

    total = len(jobs_df)

    result = (
        jobs_df.groupby("years_experience")
        .size()
        .reset_index(name='count')
        .sort_values(by='years_experience', ascending=True)
    )

    if result.empty:
        return "No experience distribution found for the given filters."

    specified = result['count'].sum()
    excluded = total - specified

    note = f"Note: {excluded} of {total} matching jobs had no explicit years-of-experience requirement mentioned and are excluded below.\n\n"
    return note + result.to_string(index=False)
