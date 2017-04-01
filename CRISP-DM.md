# Project Name

## Business Understanding

- What problem are we trying solve?
- What are the relevant metrics? How much do we plan to improve them?
- What will we deliver?

## Data Understanding

- What are the raw data sources?
- What does each 'unit' (e.g. row) of data represent?
- What are the fields (columns)?
- EDA
  - Distribution of each feature
  - Missing values
  - Distribution of target
  - Relationships between features
  - Other idiosyncracies?

## Data Preparation

- What steps are taken to prepare the data for modeling?
  - feature transformations? engineering?
  - table joins? aggregation?
- Precise description of modeling base tables.
  - What are the rows/columns of X (the predictors)?
  - What is y (the target)?

## Modeling

- What model are we using? Why?
- Assumptions?
- Regularization?

## Evaluation

- How well does the model perform?
  - Accuracy
  - ROC curves
  - Cross-validation
  - other metrics? performance?

- AB test results (if any)

## Deployment

- How is the model deployed?
  - prediction service?
  - serialized model?
  - regression coefficients?
- What support is provided after initial deployment?
