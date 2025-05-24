# Lab 5 - data & model monitoring

This lab concerns data quality monitoring, detecting data drift and model drift. They
are essential MLOps components after model deployment, extending typical monitoring
solutions to ML models.

**Learning plan**
1. Data monitoring:
   - data quality monitoring
   - univariate data drift detection
   - multivariate data drift detection
   - Evidently
2. Model monitoring:
   - model drift detection
   - confidence-based performance estimation (CBPE)
   - NannyML

**Necessary software**
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

Note that you should also activate `uv` project and install dependencies with `uv sync`.

**Lab**

See [lab instruction](LAB_INSTRUCTION.md).

**Homework**

In this lab, there is no separate homework. Laboratory instructions are long and also act
as a homework. See the [lab instruction](LAB_INSTRUCTION.pdf).

**Data**

In lab, we will use datasets:
- [Women's E-Commerce Clothing Reviews dataset](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews/data),
- [Multilingual Amazon Reviews](https://huggingface.co/datasets/defunct-datasets/amazon_reviews_multi)
