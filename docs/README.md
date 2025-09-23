# Neko Agent Documentation

This directory contains the documentation for the Neko Agent project, built with [mdBook](https://rust-lang.github.io/mdBook/).

## Local Development

To serve the documentation locally:

```bash
# Enter the docs development shell
nix develop .#docs

# From the docs/ directory
cd docs
mdbook serve --open
```

Or use the convenience command from the project root:

```bash
nix run .#docs-serve
```

## GitHub Pages Deployment

The documentation is automatically deployed to GitHub Pages via GitHub Actions when changes are pushed to the `main` branch.

### GitHub Pages Setup

To enable GitHub Pages for this repository:

1. Go to your repository settings
2. Navigate to "Pages" in the sidebar
3. Under "Source", select "GitHub Actions"
4. The workflow in `.github/workflows/docs.yml` will handle the rest

### Manual Deployment

You can also trigger a manual deployment by:

1. Going to the "Actions" tab in your GitHub repository
2. Selecting the "Deploy Documentation" workflow
3. Clicking "Run workflow"

## Structure

- `src/` - Markdown source files for the documentation
- `book.toml` - mdBook configuration
- `book/` - Generated static site (created by `mdbook build`)

## Dependencies

The documentation build requires:
- mdbook
- mdbook-mermaid (for diagrams)
- mdbook-toc (for table of contents)

These are provided by the Nix development shell.