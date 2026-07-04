# Water Network Dataset

This directory is reserved for water distribution network datasets used by the Water Network AI Analyzer.

## Dataset Requirements

The application expects structured CSV files containing operational measurements from a water distribution network.

For the 24-hour analysis and optimization workflow, the dataset must contain at least **24 rows of observations**.

Each row may represent an operational time step, such as an hourly network measurement.

## Column Naming Conventions

The current version of the application automatically identifies network variables based on column naming conventions.

| Data Category           | Naming Convention              | Example     |
| ----------------------- | ------------------------------ | ----------- |
| Pressure Reducing Valve | Column name containing `PRV`   | `PRV-01`    |
| Critical Point          | Column name starting with `J-` | `J-101`     |
| Point After Valve       | Column name ending with `-B`   | `NODE-01-B` |
| Deby / Flow Variable    | Exact column name `P-676`      | `P-676`     |

These naming conventions are used by the data validation and preprocessing pipeline to automatically organize the dataset into analytical categories.

## Example Dataset Structure

```text
PRV-01,PRV-02,J-101,J-102,NODE-01-B,NODE-02-B,P-676
35.2,37.1,28.4,30.1,32.5,34.0,120.5
34.8,36.9,27.9,29.8,32.1,33.7,118.9
36.0,37.5,28.8,30.5,33.0,34.4,122.1
...
```

The values shown above are illustrative examples only and do not represent a real water distribution network.

## Data Preprocessing

After loading a compatible CSV file, the application performs a preprocessing workflow that includes:

* Dataset structure validation
* Automatic network variable identification
* Numeric data processing
* Missing-value imputation using K-Nearest Neighbors
* Interquartile Range based outlier filtering
* Feature preparation for machine learning workflows

Feature scaling is applied during XGBoost model training.

## Data Privacy

Operational water infrastructure datasets may contain sensitive engineering or infrastructure information.

For this reason, real operational datasets used during development are not included in this public repository.

Users should only upload datasets they are authorized to process and share.

## Sample Data

A synthetic or anonymized sample dataset may be added to this directory for demonstration and testing purposes.

Future versions of the project are intended to support configurable schema mapping, reducing dependency on fixed column naming conventions.
