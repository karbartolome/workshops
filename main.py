"""Builds the repo's landing page (index.html) listing all Quarto revealjs slide decks.

Usage:
    python main.py [--no-render]

It scans the ``slides/`` folder for rendered slide decks (``slides*.html``),
merges them with metadata from ``slides_descriptions.json`` (adding skeleton
entries for decks that aren't described yet), generates ``index.qmd`` with a
card grid preview of each deck, renders it to ``index.html`` with the Quarto
CLI, and (re)writes ``404.html`` so that old pre-migration URLs (when every
workshop folder lived at the repo root) keep redirecting to their new home
under ``slides/``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLIDES_DIR = ROOT / "slides"
DESCRIPTIONS_PATH = ROOT / "metadata" / "slides_descriptions.json"
INDEX_QMD_PATH = ROOT / "index.qmd"
TAGS_CATEGORIES_PATH = ROOT / "metadata" / "tags_categories.json"
NOT_FOUND_PATH = ROOT / "404.html"

# Workshop folders that used to live at the repo root (before they moved into
# slides/) mapped to their rendered deck filename. Used to build redirects so
# old links keep working.
LEGACY_SLUGS = {
    "20240513-uba-calibracion": "slides.html",
    "20250905-uba-pipelines": "slides.html",
}
DEFAULT_TAG_COLORS = [
    "#107895",
    "#7a5cff",
    "#2a9d8f",
    "#f4a261",
    "#e76f51",
    "#6c5ce7",
    "#16a085",
    "#d35400",
    "#e84393",
    "#00b894",
]

# Subfolder names to skip while looking for a rendered deck inside a workshop folder.
EXCLUDED_SUBDIRS = {"_extensions", "artifacts", "catboost_info", "notebooks", ".quarto"}


def find_slide_html(folder: Path) -> Path | None:
    """Return the most likely rendered slide deck inside a folder."""
    candidates = [
        p
        for p in folder.rglob("*.html")
    if (
      (
        p.name.lower() == "slides.html"
        or (p.name.lower().startswith("slides") and "short" not in p.name.lower())
        or p.name.lower() == "index.html"
      )
      and not EXCLUDED_SUBDIRS.intersection(p.relative_to(folder).parts[:-1])
    )
    ]
    if not candidates:
        return None
    preferred = ("slides.html", "index.html")
    candidates.sort(key=lambda p: (preferred.index(p.name) if p.name in preferred else len(preferred), len(p.relative_to(folder).parts), p.name))
    return candidates[0]


def discover_decks() -> dict[str, Path]:
    """Map workshop folder name -> rendered slide html path (relative to ROOT)."""
    decks = {}
    if not SLIDES_DIR.exists():
        return decks

    for folder in sorted(SLIDES_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        html_path = find_slide_html(folder)
        if html_path is not None:
            decks[folder.name] = html_path.relative_to(ROOT)
    return decks


def load_descriptions() -> dict:
    if DESCRIPTIONS_PATH.exists():
        return json.loads(DESCRIPTIONS_PATH.read_text(encoding="utf-8"))
    return {}


def sync_descriptions(descriptions: dict, decks: dict[str, Path]) -> dict:
    """Add skeleton entries for newly discovered decks, without touching existing ones."""
    changed = False
    for name in decks:
        if name not in descriptions:
            descriptions[name] = {
                "name": name,
                "date": "",
                "description": "",
                "tags": [],
            }
            changed = True
    if changed:
        DESCRIPTIONS_PATH.write_text(
            json.dumps(descriptions, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Updated {DESCRIPTIONS_PATH.name} with new deck(s).")
    return descriptions


def sanitize_css_token(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-") or "tag"


def load_tag_categories() -> tuple[dict[str, str], dict[str, str]]:
    tag_to_category: dict[str, str] = {}
    category_colors: dict[str, str] = {}

    if not TAGS_CATEGORIES_PATH.exists():
        return tag_to_category, category_colors

    try:
        raw = json.loads(TAGS_CATEGORIES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return tag_to_category, category_colors

    for idx, (category, payload) in enumerate(raw.items()):
        if isinstance(payload, dict):
            tags = payload.get("tags", [])
            color = payload.get("color") or DEFAULT_TAG_COLORS[idx % len(DEFAULT_TAG_COLORS)]
        else:
            tags = payload if isinstance(payload, list) else []
            color = DEFAULT_TAG_COLORS[idx % len(DEFAULT_TAG_COLORS)]

        category_colors[category] = color
        for tag in tags:
            tag_to_category[tag.lower()] = category
    return tag_to_category, category_colors


def tag_pills(tags: list[str], tag_to_category: dict[str, str], category_colors: dict[str, str]) -> str:
    pills = []
    for tag in tags:
        category = tag_to_category.get(tag.lower(), "general")
        color = category_colors.get(category, DEFAULT_TAG_COLORS[0])
        pills.append(
            f'<span class="tag tag-{sanitize_css_token(category)}" '
            f'style="--tag-bg: {color}; --tag-fg: #fff;">{tag}</span>'
        )
    return "".join(pills)


def render_card(
    name: str,
    meta: dict,
    href: str,
    tag_to_category: dict[str, str],
    category_colors: dict[str, str],
) -> str:
    title = meta.get("name") or name
    date = meta.get("date", "")
    description = meta.get("description", "")
    tags = meta.get("tags", [])
    tags_html = tag_pills(tags, tag_to_category, category_colors)
    data_tags = "|".join(t.lower() for t in tags)
    return f"""
<a class="card" href="{href}" target="_blank" rel="noopener" data-tags="{data_tags}">
  <div class="thumb-wrapper">
    <iframe class="thumb-iframe" src="{href}" scrolling="no" tabindex="-1"></iframe>
  </div>
  <div class="card-body">
    <h3>{title}</h3>
    <p class="date">{date}</p>
    <p class="desc">{description}</p>
    <div class="tags">{tags_html}</div>
  </div>
</a>"""


def render_filter_bar(decks: dict, descriptions: dict, tag_to_category: dict[str, str], category_colors: dict[str, str]) -> str:
    all_tags = sorted(
        {tag for name in decks for tag in descriptions.get(name, {}).get("tags", [])},
        key=str.lower,
    )
    buttons = []
    for tag in all_tags:
        category = tag_to_category.get(tag.lower(), "general")
        color = category_colors.get(category, DEFAULT_TAG_COLORS[0])
        buttons.append(
            f'<button class="tag-filter tag-filter-{sanitize_css_token(category)}" '
            f'data-tag="{tag.lower()}" style="--tag-bg: {color}; --tag-fg: #fff;">{tag}</button>'
        )
    return f"""
<div class="filter-bar">
  <input type="text" id="tag-search" placeholder="Buscar tags..." />
  <div class="tag-filter-list" id="tag-filter-list">{''.join(buttons)}</div>
  <button class="clear-filters" id="clear-filters">Limpiar filtros</button>
</div>"""


FILTER_JS = """
<script>
(function () {
  const searchInput = document.getElementById('tag-search');
  const tagButtons = Array.from(document.querySelectorAll('.tag-filter'));
  const clearBtn = document.getElementById('clear-filters');
  const cards = Array.from(document.querySelectorAll('.card'));
  const activeTags = new Set();

  function applyFilter() {
    cards.forEach((card) => {
      const cardTags = (card.dataset.tags || '').split('|').filter(Boolean);
      const visible = activeTags.size === 0 || Array.from(activeTags).every((t) => cardTags.includes(t));
      card.style.display = visible ? '' : 'none';
    });
  }

  tagButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const tag = btn.dataset.tag;
      if (activeTags.has(tag)) {
        activeTags.delete(tag);
        btn.classList.remove('active');
      } else {
        activeTags.add(tag);
        btn.classList.add('active');
      }
      applyFilter();
    });
  });

  searchInput.addEventListener('input', () => {
    const query = searchInput.value.trim().toLowerCase();
    tagButtons.forEach((btn) => {
      btn.style.display = btn.dataset.tag.includes(query) ? '' : 'none';
    });
  });

  clearBtn.addEventListener('click', () => {
    activeTags.clear();
    tagButtons.forEach((btn) => btn.classList.remove('active'));
    applyFilter();
  });
})();
</script>"""


CARD_CSS = """
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 28px;
  margin-top: 1.5em;
}
.card {
  display: block;
  border: 1px solid #e2e2e2;
  border-radius: 8px;
  overflow: hidden;
  text-decoration: none;
  color: #131516;
  background: #fff;
  transition: box-shadow 0.15s ease, transform 0.15s ease;
}
.card:hover {
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.12);
  transform: translateY(-2px);
}
.thumb-wrapper {
  position: relative;
  width: 100%;
  padding-top: 56.25%; /* 16:9 */
  overflow: hidden;
  background: #131516;
  border-bottom: 1px solid #e2e2e2;
}
.thumb-iframe {
  position: absolute;
  top: 0;
  left: 0;
  width: 400%;
  height: 400%;
  transform: scale(0.25);
  transform-origin: top left;
  border: none;
  pointer-events: none;
}
.card-body {
  padding: 14px 16px 18px;
}
.card-body h3 {
  margin: 0 0 4px;
  font-size: 1.05rem;
  color: #131516;
}
.card-body .date {
  margin: 0 0 8px;
  font-size: 0.85rem;
  color: #6b6b6b;
}
.card-body .desc {
  margin: 0 0 10px;
  font-size: 0.9rem;
  color: #333;
}
.tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.tag {
  font-size: 0.75rem;
  padding: 2px 9px;
  border-radius: 999px;
  background: var(--tag-bg, #107895);
  color: var(--tag-fg, #fff);
}
.filter-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  padding: 14px 16px;
  border: 1px solid #e2e2e2;
  border-radius: 8px;
  background: #f7f7f7;
}
#tag-search {
  padding: 6px 10px;
  border: 1px solid #ccc;
  border-radius: 999px;
  font-size: 0.85rem;
  min-width: 180px;
}
.tag-filter-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  flex: 1;
}
.tag-filter {
  font-size: 0.8rem;
  padding: 4px 12px;
  border-radius: 999px;
  border: 1px solid var(--tag-bg, #107895);
  background: #fff;
  color: var(--tag-bg, #107895);
  cursor: pointer;
}
.tag-filter:hover {
  background: rgba(17, 24, 39, 0.04);
}
.tag-filter.active {
  background: var(--tag-bg, #107895);
  color: var(--tag-fg, #fff);
}
.clear-filters {
  font-size: 0.8rem;
  padding: 4px 12px;
  border-radius: 999px;
  border: 1px solid #9a2515;
  background: #fff;
  color: #9a2515;
  cursor: pointer;
}
.clear-filters:hover {
  background: #fbeae7;
}
"""


def write_not_found_page() -> None:
    """Write 404.html so pre-migration URLs redirect to their new slides/ location.

    Matching is done on the workshop slug anywhere in the URL path (instead of
    a fixed prefix), so the redirect works whether the site is served from a
    custom domain or from a GitHub Pages project path like /workshops/.
    """
    legacy_json = json.dumps(LEGACY_SLUGS)
    redirect_script = f"""
<script>
(() => {{
  const legacy = {legacy_json};
  const path = window.location.pathname;
  for (const [slug, file] of Object.entries(legacy)) {{
    const idx = path.indexOf("/" + slug);
    if (idx !== -1) {{
      const prefix = path.slice(0, idx);
      const target = `${{prefix}}/slides/${{slug}}/${{file}}`;
      window.location.replace(target + window.location.search + window.location.hash);
      break;
    }}
  }}
}})();
</script>
""".strip()
    NOT_FOUND_PATH.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        + redirect_script + "</head><body></body></html>",
        encoding="utf-8",
    )


def build_qmd(decks: dict[str, Path], descriptions: dict) -> str:
    def sort_key(name: str):
        return descriptions.get(name, {}).get("date", "") or "0000-00-00"

    tag_to_category, category_colors = load_tag_categories()
    cards = "\n".join(
        render_card(name, descriptions.get(name, {}), decks[name].as_posix(), tag_to_category, category_colors)
        for name in sorted(decks, key=sort_key, reverse=True)
    )
    filter_bar = render_filter_bar(decks, descriptions, tag_to_category, category_colors)

    return f"""---
title: "Workshops"
subtitle: "Karina Bartolomé"
format:
  html:
    page-layout: full
    css: []
    theme: cosmo
    grid:
      body-width: 1100px
---

```{{=html}}
<style>
{CARD_CSS}
</style>
```

```{{=html}}
{filter_bar}
<div class="grid">
{cards}
</div>
{FILTER_JS}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-render", action="store_true", help="Only regenerate index.qmd, skip quarto render."
    )
    args = parser.parse_args()

    decks = discover_decks()
    if not decks:
        print("No rendered slide decks (slides*.html) found.", file=sys.stderr)
        sys.exit(1)

    descriptions = sync_descriptions(load_descriptions(), decks)
    write_not_found_page()
    INDEX_QMD_PATH.write_text(build_qmd(decks, descriptions), encoding="utf-8")
    print(f"Wrote {INDEX_QMD_PATH.name} with {len(decks)} deck(s).")

    if not args.no_render:
        subprocess.run(["quarto", "render", str(INDEX_QMD_PATH)], cwd=ROOT, check=True)
        print("Rendered index.html.")


if __name__ == "__main__":
    main()
