# Dataset Methodology & Limitations

## Data Collection
Job postings are collected from the JobTech API using a set of predefined search terms
targeting AI and Data roles in Sweden (e.g. "data engineer", "machine learning", "ai engineer"). 
The dataset reflects a snapshot of postings available at collection time, not real-time 
current openings.

## Skill Detection
Skill detection is based on predefined keyword and regex patterns. Tracked skills include
examples such as Python, SQL, PySpark, Spark, Databricks, Airflow, Azure, AWS, GCP, Snowflake,
dbt, and Power BI. Job descriptions may mention skills in ways not yet captured by these
patterns, so skill counts should be read as detected mentions rather than a perfectly
exhaustive classification.

## Workplace Type Classification
Workplace type classification is keyword-based. Descriptions containing words such as
"remote", "distans", or "hybrid" are classified accordingly. The `not_specified` category
means no remote or hybrid keyword was detected in the posting — it does not necessarily mean
the role is fully on-site, only that workplace type wasn't explicitly mentioned in the text.

## Experience Requirements
Explicit years-of-experience requirements are extracted from job descriptions using regex
patterns. Blank/missing values mean no explicit year requirement was detected in the text —
this does not mean no experience is required, only that a specific number of years wasn't stated.

## Market Observations
Across the dataset, Python and SQL are the two most consistently requested skills, appearing
at the top of the list for both Data Engineer and Machine Learning roles, though with very
different volumes — data engineering postings mention specific tools like SQL, Databricks,
and dbt far more often, reflecting a larger and more mature job segment compared to the
smaller pool of dedicated ML postings.

Cloud platform experience is in high demand across roles, with Azure appearing most
frequently, followed by AWS and GCP — likely reflecting the mix of enterprise, public-sector,
and startup employers represented in the dataset.

Most postings do not explicitly state a workplace type: the majority fall into the
`not_specified` category, with a much smaller share explicitly advertised as hybrid, and an
even smaller share as fully remote. Job postings are heavily concentrated in Stockholm, with
Gothenburg, Malmö, and Lund also well represented — reflecting Sweden's general concentration
of tech employers in and around its largest cities.

## General Limitations
- Skill, workplace type, and experience detection rely on keyword/regex patterns and should
  be interpreted as detected mentions, not perfect classifications.
- The dataset depends on the selected search terms and what JobTech API returns for them —
  it is not an exhaustive census of all Swedish AI/Data job postings.
- Employer counts may be affected by consultancies, duplicate-like postings, and broad search
  terms that pull in loosely related roles.