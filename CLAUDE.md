# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal blog/website built with [Zola](https://www.getzola.org/) static site generator using the [serene](https://github.com/isunjn/serene) theme (v5.5.0). Hosted on GitHub Pages at `https://yashrb24.github.io`.

## Development Commands

```sh
# Start local dev server with live reload
zola serve

# Build site (outputs to public/)
zola build

# Include draft posts
zola serve --drafts
zola build --drafts
```

## Theme Management

The theme is a git submodule at `themes/serene/`. Do NOT modify theme files directly.

```sh
# After fresh clone
git submodule update --init --recursive

# Update theme
git submodule update --remote themes/serene
```

## Content Structure

- `content/_index.md` - Homepage
- `content/about/_index.md` - About page
- `content/posts/` - Blog posts
- `content/projects/` - Projects (uses `projects.toml` for data)
- `content/experience/` - Experience (uses `experience.toml` for data)
- `content/publications/` - Publications (uses `publications.toml` for data)

## Adding Content

### Blog Post
Create `content/posts/post-slug.md`:
```toml
+++
title = "Post Title"
description = "Brief description"
date = 2025-01-01

[taxonomies]
tags = ["tag1", "tag2"]

[extra]
toc = true
+++

Content here...
```

### Data-Driven Sections
Projects, experience, and publications use TOML data files (e.g., `projects.toml`) instead of individual markdown files.

## Key Configuration

- Main config: `config.toml`
- Custom templates: `templates/` (override theme templates)
- Custom styles: `sass/` (override theme SCSS)
- Static assets: `static/` (images, favicon in `static/img/`)

## Theme Documentation

Full theme usage: https://github.com/isunjn/serene/blob/latest/USAGE.md
