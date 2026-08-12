# The Next Word Problem: the code behind every chart

Every chart on the four hour decks of **The Next Word Problem**, ASE Summer
School 2026, published as a Quantlet: one folder per chart, holding the code
that draws it, the data it reads, a `Metainfo.txt` describing it and the course
palette. Each folder runs on its own.

    cd NextWord_AttentionHeatmap
    python NextWord_AttentionHeatmap.py

The figure appears in `figures/`.

## These folders are generated, and checked

The code in each folder is not a second implementation of the chart. It is the
course generator's own source, extracted with two lines added: one that makes
relative paths resolve inside the folder, and one that creates `figures/`,
because matplotlib will not create the directory it is told to save into.

That claim is measured rather than asserted. The course build runs all 2
folders and requires each figure to come out byte-identical to the one printed
on the slide.

Because they are generated, a change made here is lost on the next publish.
Corrections belong in the course repository.

## Course site

The decks, the exercises, the reference sheet and the lab notebooks are at
https://josterri.github.io/2025_ASE_SummerSchool/

## The 2 charts

| Quantlet | Chart |
| --- | --- |
| [`NextWord_AttentionHeatmap`](NextWord_AttentionHeatmap) | Attention heatmap visualization |
| [`NextWord_VocabGrowth`](NextWord_VocabGrowth) | Possible n-grams explode; the corpus cannot keep up. |
