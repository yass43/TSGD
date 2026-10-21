# TSGD paper template

This directory is a modular, heading-only LaTeX paper skeleton based on the agreed
paper outline. It contains no manuscript prose, claims, equations, or citations.

## Build

From this directory, run:

```sh
latexmk -pdf main.tex
```

Or run `make`. To remove generated files, run `latexmk -C` or `make clean`.
The project uses standard TeX Live/MiKTeX packages and requires no custom venue class.

## Layout

- `main.tex`: document assembly, title, and author placeholders.
- `preamble.tex`: packages and document-wide formatting.
- `macros.tex`: project-specific notation.
- `theorems.tex`: theorem-like environments.
- `sections/`: section and subsection headings with empty bodies.
- `appendix/`: appendix headings with empty bodies.
- `references.bib`: empty bibliography database.
- `figures/`: figure assets.
