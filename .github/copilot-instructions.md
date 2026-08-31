# Copilot Instructions

## Repository

Quarto RevealJS slides written in Python.
Sometimes notebooks are used.

## Language

* All documentation, docstrings, and code comments must be in **Spanish**.
* Variables, functions, classes, and other code identifiers must be in **English**.
* Slide content should be in Spanish.

## DataFrames

* DataFrame variables must always follow `df_{something}`.
* Functions receiving a DataFrame must name the parameter `df`.
* The first operation inside every function receiving `df` must be:

  ```python
  data = df.copy()
  ```
* Never modify the input DataFrame directly.

## Pandas

Prefer readable method chaining / piping logic:

```python
(
    df_users
    .groupby("segment", as_index=False)
    .agg(mean_score=("score", "mean"))
    .assign(score_pct=lambda data: data["mean_score"] * 100)
)
```

Prefer pipelines over unnecessary intermediate DataFrames.

## Code Style

* Prioritize clarity and simplicity.
* Avoid unnecessary comments. Code should be self-explanatory.
* Avoid unnecessary abstractions.
* Use descriptive English names.
* Keep functions small and focused.
* Ensure code adheres to PEP 8 standards.
* Write docstrings for all functions following the NumPy/SciPy style.
