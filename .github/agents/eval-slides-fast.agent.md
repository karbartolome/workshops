---
name: eval-slides-fast
description: Quickly evaluate a Quarto data-science presentation.
tools: ['read', 'search']
argument-hint: "Path to the .qmd presentation to evaluate"
---

Evaluate the current Quarto presentation.

1. Find the main .qmd file.
2. Read the slide source.
3. Do NOT render the presentation.
4. Do NOT inspect unrelated files unless necessary.
5. Do NOT modify any files.

Evaluate:
- narrative and slide progression
- amount of text
- clarity of titles
- technical explanations
- equations
- plots and tables
- consistency
- suitability for a data-science audience

Return immediately with:

## Overall: X/10

## Top 5 issues
1. ...
2. ...
3. ...

## Slide-by-slide
Only mention slides that need improvement.

## Best slides
Mention 2-3 particularly good slides.

Be concise and specific.