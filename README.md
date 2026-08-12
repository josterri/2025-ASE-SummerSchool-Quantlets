# The Next Word Problem: the code behind every chart

Every chart on the four hour decks of **The Next Word Problem**, ASE Summer
School 2026, published as a Quantlet: one folder per chart, holding the code
that draws it, the data it reads, a `Metainfo.txt` describing it and the course
palette. Each folder runs on its own.

    cd NextWord_AgenticRiskSurface
    python NextWord_AgenticRiskSurface.py

The figure appears in `figures/`.

## These folders are generated, and checked

The code in each folder is not a second implementation of the chart. It is the
course generator's own source, extracted with two lines added: one that makes
relative paths resolve inside the folder, and one that creates `figures/`,
because matplotlib will not create the directory it is told to save into.

That claim is measured rather than asserted. The course build runs all 28
folders and requires each figure to come out byte-identical to the one printed
on the slide.

Because they are generated, a change made here is lost on the next publish. If
something looks wrong, open an issue on this repository and the fix will be made
where the folders come from.

## Course site

The decks, the exercises, the reference sheet and the lab notebooks are at
https://josterri.github.io/2025_ASE_SummerSchool/

## The 28 charts

| Quantlet | Chart |
| --- | --- |
| [`NextWord_AgenticRiskSurface`](NextWord_AgenticRiskSurface) | Exposed surface widens as autonomy grows, structural, not quantitative |
| [`NextWord_ApplicationsTree`](NextWord_ApplicationsTree) | Applications tree diagram |
| [`NextWord_AttentionHeatmap`](NextWord_AttentionHeatmap) | Attention heatmap visualization |
| [`NextWord_BERTVsGPT`](NextWord_BERTVsGPT) | BERT bidirectional mask vs GPT causal mask |
| [`NextWord_ComputeRequirements`](NextWord_ComputeRequirements) | Compute requirements graph |
| [`NextWord_CostQualityLatency`](NextWord_CostQualityLatency) | Cost vs quality trade-off across model classes, bubble size is latency (indicative) |
| [`NextWord_EmergentAbilities`](NextWord_EmergentAbilities) | Emergent abilities chart |
| [`NextWord_FinetuningVsPrompting`](NextWord_FinetuningVsPrompting) | Fine-tuning vs prompting comparison |
| [`NextWord_InferenceComputeScaling`](NextWord_InferenceComputeScaling) | Inference-time compute scaling (indicative) |
| [`NextWord_LSTMGates`](NextWord_LSTMGates) | The cell state as a highway; gates are learned taps on it. |
| [`NextWord_MMLUSaturation`](NextWord_MMLUSaturation) | Benchmark saturation and replacement (indicative) |
| [`NextWord_MarkovContextWindow`](NextWord_MarkovContextWindow) | How much of the sentence each generation is allowed to see. |
| [`NextWord_MultiheadPatterns`](NextWord_MultiheadPatterns) | Different heads specialise: local, syntax, semantic, broad context. |
| [`NextWord_MultimodalFusion`](NextWord_MultimodalFusion) | Separate encoders converge on one shared space, then a single decoder |
| [`NextWord_OneHotVsDense`](NextWord_OneHotVsDense) | Contrast a one-hot vector against a dense embedding. |
| [`NextWord_PerplexityLadder`](NextWord_PerplexityLadder) | Perplexity is 2 to the power of the entropy on the generation card. |
| [`NextWord_PositionalEncoding`](NextWord_PositionalEncoding) | Proper heatmap of the sinusoidal positional encoding, replacing the hand-drawn tikz grid. |
| [`NextWord_PretrainFinetuneFlow`](NextWord_PretrainFinetuneFlow) | Pre-train once, adapt many times |
| [`NextWord_QKVFlow`](NextWord_QKVFlow) | One token, three learned projections, scaled dot-product attention. |
| [`NextWord_RAGPipeline`](NextWord_RAGPipeline) | RAG pipeline. The retrieval loop stays visually distinct from a plain LLM call |
| [`NextWord_RLHFPipeline`](NextWord_RLHFPipeline) | RLHF in three stages left to right, human preference data enters at stage (b) only |
| [`NextWord_RNNMemoryDecay`](NextWord_RNNMemoryDecay) | RNN memory decay visualization |
| [`NextWord_SamplingStrategies`](NextWord_SamplingStrategies) | Greedy, temperature, top-k and nucleus sampling reshape one distribution |
| [`NextWord_ScalingLaws`](NextWord_ScalingLaws) | Scaling laws visualization |
| [`NextWord_TokenizationExample`](NextWord_TokenizationExample) | Subword tokenization example |
| [`NextWord_ToyCorpusDistribution`](NextWord_ToyCorpusDistribution) | Generation 3: what the 4-gram model actually predicts for the blank. |
| [`NextWord_VocabGrowth`](NextWord_VocabGrowth) | Possible n-grams explode; the corpus cannot keep up. |
| [`NextWord_WordVectorSpace`](NextWord_WordVectorSpace) | Word vector space visualization |
