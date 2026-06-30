# assistant/tools.py
import pandas as pd
from pathlib import Path
from langchain_core.tools import tool

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

@tool
def get_top_skills_tool(role: str = None, region: str = None, workplace_type: str = None, top_n: int = 10) -> str:
    """
    Get the most frequently requested skills in Swedish job postings for positions within AI/Data.

    Use this when the user asks about which skills are common, popular,
    or in-demand. You can optionally filter by job role/title (e.g. "data engineer"),
    by region (e.g. "Stockholms län", "Västra Götalands län"), or by workplace type ("remote", "hybrid", "not_specified").

    Args:
        role: Filter to job titles containing this text (case-insensitive). e.g. "data engineer"
        region: Filter to jobs in this region (case-insensitive). e.g. "Stockholms län"
        workplace_type: Filter by workplace type. One of "remote", "hybrid", "not_specified"
        top_n: How many top skills to return (default 10)
    """
    return get_top_skills(role=role, region=region, workplace_type=workplace_type, top_n=top_n)


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

@tool
def get_top_employers_tool(skill: str = None, region: str = None, top_n: int = 10) -> str:
    """
    Get the employers with the most published job postings for positions within AI/Data in Sweden,
    based on the collected dataset, and how many postings they have.

    Use this when the user asks about which employers have published many job postings or
    appear most often in the dataset for AI/Data roles. You can optionally filter
    by skill (e.g. "python", "sql", "azure"), meaning skills that are highly valuable within that
    sector. You can also filter by region (e.g. "Stockholms län", "Västra Götalands län", "Skåne län" etc.).

    Note: this reflects a snapshot of collected job postings, not real-time current openings.

    Args:
        skill: Filter to skills containing this text (case-insensitive). e.g. "python", "sql"
        region: Filter to jobs in this region (case-insensitive). e.g. "Stockholms län", "Västra Götalands län"
        top_n: How many top employers to return (default 10)
    """
    return get_top_employers(skill=skill, region=region, top_n=top_n)


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

@tool
def get_top_locations_tool(skill: str = None, workplace_type: str = None, top_n: int = 10) -> str:
    """
    Get the regions and municipalities with the most job postings in the dataset,
    based on the collected dataset.

    Use this when the user asks which locations or cities have the most AI/Data job
    postings in Sweden, or where job opportunities are concentrated geographically.
    You can optionally filter by skill (e.g. "python", "sql", "azure") to see where
    jobs requiring that skill are concentrated, or by workplace type.

    Workplace types: "hybrid" (partly remote, partly office), "remote" (primarily
    work from home), or "not_specified" (no workplace type mentioned in the posting).

    Note: this reflects a snapshot of collected job postings, not real-time current openings.

    Args:
        skill: Filter to skills containing this text (case-insensitive). e.g. "python", "sql"
        workplace_type: One of "hybrid", "remote", or "not_specified"
        top_n: How many top locations to return (default 10)
    """
    return get_top_locations(skill=skill, workplace_type=workplace_type, top_n=top_n)

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

@tool
def get_workplace_type_distribution_tool(region: str = None, skill: str = None) -> str:
    """
    Get the breakdown of job postings by workplace type (remote, hybrid, or not
    specified) in the dataset.

    Use this when the user asks how common remote or hybrid work is in the AI/Data
    sector in Sweden, or wants a distribution/breakdown of workplace arrangements.
    You can optionally filter by region or by skill to see workplace type patterns
    within a specific area or for a specific skill.

    Workplace types: "remote" (primarily work from home), "hybrid" (partly remote,
    partly office), or "not_specified" (no workplace type mentioned in the posting —
    this does not necessarily mean the role is fully on-site, just that it wasn't
    mentioned in the text).

    Note: this reflects a snapshot of collected job postings, not real-time current openings.

    Args:
        region: Filter to jobs in this region (case-insensitive). e.g. "Stockholms län"
        skill: Filter to jobs requiring this skill (case-insensitive). e.g. "python"
    """
    return get_workplace_type_distribution(region=region, skill=skill)


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

@tool
def get_experience_distribution_tool(role: str = None, region: str = None) -> str:
    """
    Get the distribution of explicit years-of-experience requirements mentioned in
    job postings, e.g. how many postings ask for 2 years, 5 years, etc.

    Use this when the user asks what experience level is typical or common for a
    role, or wants to understand seniority requirements in the AI/Data job market.
    You can optionally filter by job role/title or by region.

    Important: only a minority of job postings state an explicit number of years
    of experience required. The result will include a note showing how many matching
    postings had no explicit requirement and were excluded from the breakdown — make
    sure to mention this limitation when answering the user, since the numbers only
    describe the subset of postings that specified a number.

    Args:
        role: Filter to job titles containing this text (case-insensitive). e.g. "data engineer"
        region: Filter to jobs in this region (case-insensitive). e.g. "Stockholms län"
    """
    return get_experience_distribution(role=role, region=region)