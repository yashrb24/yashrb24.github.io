+++
title = "[draft] What does the temporal objective in Temporal SAEs learn? A replication and characterization"
description = "A replication of the Temporal Sparse Autoencoder and a characterization of what its contrastive objective changes, when it helps, and where its boundaries are"
date = 2026-07-19

[taxonomies]
tags = ["interpretability", "ai-safety", "sparse-autoencoders", "machine-learning"]

[extra]
toc = true
comment = false
math = true
+++

> **Note (draft):** This is a machine-assisted port of the source artifact and needs major rewrites before publishing, tighten the prose into my own voice, verify every number against the actual runs, the four figures are inlined below as self-contained SVG from the source artifact (re-render them if you prefer raster). Cut or restructure sections as needed.

This is a draft writeup from my BlueDot Impact project sprint: we replicated the Temporal Sparse Autoencoder, then mapped what its contrastive objective changes in the learned features, when it improves semantic readability, and what its boundary conditions are. The headline effect replicates to the third decimal place; its downstream benefit is real, bounded, and traceable to a specific mechanism. Everything below is on **Pythia-160m, layer 8, one SAE configuration**, run on one RTX 4500 Ada. Draft for review, figures and framing may change.

> **Epistemic status.** Confident about the results reported here. Everything load-bearing is seed-replicated and paired with matched controls, and the strongest claims are the ones backed by exact-zero results under held-out-sequence evaluation. Less confident about generalization to larger models: the paper's headline probing figure is on Gemma2-2b, which we have not run, and the mechanism we identify predicts the semantic benefit should be *larger* there. The study is two-sided by design: some of the paper's results replicate and strengthen under our controls; others turn out to be bounded by measurement or data conditions.
>
> Subject paper: Bhalla, Oesterling, Mayrink Verduna, Lakkaraju & Calmon, *Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability*, ICLR 2026 ([arXiv:2511.05541](https://arxiv.org/abs/2511.05541)). Run on the authors' released code; all extensions subclass their trainer with the upstream untouched.

<style>
.tsae-fig{--surface:#fbfcfd;--ink-soft:#48505a;--ink-faint:#6c757e;--rule:#d5dade;--rule-soft:#e4e8eb;--ch1:#2a78d6;--ch2:#1baf7a;--ch3:#eda100;margin:1.8rem 0;}
body.dark .tsae-fig{--surface:#161a1f;--ink-soft:#a4adb6;--ink-faint:#7d8791;--rule:#2a3138;--rule-soft:#20262c;--ch1:#3987e5;--ch2:#199e70;--ch3:#c98500;}
.tsae-fig .chart-box{background:var(--surface);border:1px solid var(--rule);border-radius:5px;padding:.9rem .5rem .3rem;overflow-x:auto;-webkit-overflow-scrolling:touch;}
.tsae-fig .chart{display:block;width:100%;height:auto;min-width:540px;}
.tsae-fig .chart text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:11px;fill:var(--ink-faint);}
.tsae-fig .chart .ptitle{font-size:11.5px;font-weight:650;fill:var(--ink-soft);}
.tsae-fig .chart .dlabel{font-size:11px;font-weight:600;fill:var(--ink-soft);}
.tsae-fig .chart .vlabel{font-size:10.5px;font-weight:600;fill:var(--ink-soft);}
.tsae-fig .chart .grid{stroke:var(--rule-soft);stroke-width:1;}
.tsae-fig .chart .axis{stroke:var(--rule);stroke-width:1;}
.tsae-fig .chart .refline{stroke:var(--ink-faint);stroke-width:1.4;stroke-dasharray:5 4;fill:none;}
.tsae-fig .chart .zline{stroke:var(--rule);stroke-width:1;stroke-dasharray:2 3;}
.tsae-fig .chart .l1{stroke:var(--ch1);stroke-width:2;fill:none;}
.tsae-fig .chart .l2{stroke:var(--ch2);stroke-width:2;fill:none;}
.tsae-fig .chart .l3{stroke:var(--ch3);stroke-width:2;fill:none;}
.tsae-fig .chart .ln{stroke:var(--ink-soft);stroke-width:2;fill:none;}
.tsae-fig .chart .p1{fill:var(--ch1);stroke:var(--surface);stroke-width:1.5;}
.tsae-fig .chart .p2{fill:var(--ch2);stroke:var(--surface);stroke-width:1.5;}
.tsae-fig .chart .p3{fill:var(--ch3);stroke:var(--surface);stroke-width:1.5;}
.tsae-fig .chart .pn{fill:var(--ink-soft);stroke:var(--surface);stroke-width:1.5;}
.tsae-fig .chart .err{stroke:var(--ch1);stroke-width:1.2;opacity:.55;}
.tsae-fig .chart .bar{fill:var(--ch1);}
.tsae-fig .legend{display:flex;flex-wrap:wrap;gap:.4rem 1.2rem;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:.78rem;color:var(--ink-soft);margin:.55rem .3rem .1rem;}
.tsae-fig .legend .sw{display:inline-block;width:15px;height:3px;border-radius:2px;vertical-align:middle;margin-right:.42rem;}
.tsae-fig .legend .sw.s1{background:var(--ch1);}
.tsae-fig .legend .sw.s2{background:var(--ch2);}
.tsae-fig .legend .sw.s3{background:var(--ch3);}
.tsae-fig .legend .sw.sn{background:var(--ink-soft);}
.tsae-fig .legend .sw.sref{background:none;border-top:2px dashed var(--ink-faint);height:0;}
</style>

## 1. The paper's idea, and the question we set out to characterize

Sparse autoencoders (SAEs) have a persistent limitation: trained only to reconstruct, they tend to learn shallow, token-level features, "this is a period," "this is the token `the`", rather than the high-level semantic concepts we would most like to read off a model. The Temporal SAE (T-SAE) paper starts from a linguistic observation and turns it into a training signal: **meaning changes slowly**. The topic of a paragraph does not flip from word to word even though the tokens do, so if some features are encouraged to change slowly across a sequence, those slow features should end up carrying the meaning.

The paper implements this with a contrastive term added to SAE training that encourages a designated "high-level" group of features at each token to resemble the same features one token earlier. As evidence, it reports two results side by side: a **smoothness gap**, high-level features vary more slowly over a sequence than low-level ones (+0.08 in the paper's units), and probing gains, with linear probes reading semantics and "context" more accurately from the high-level features than from a plain SAE's.

We want to be upfront about why this method earned a careful study. The effect is real, replicates essentially exactly (§3), and the idea, importing a slowness prior from slow feature analysis and contrastive representation learning into SAE training, is a promising direction for making SAE features less token-bound. A result this clean deserves careful characterization, and the interesting scientific questions begin exactly where the paper stops. Its argument couples two claims that can be studied separately:

- **(A) Does the objective make the high-level features slower?** This is what the smoothness gap measures.
- **(B) Does that slowness make the features more useful downstream?** This is the motivating promise.

The paper demonstrates (A) convincingly, and its §3.1 modeling assumption identifies "slowly-varying" with "semantic" by definition, so (B) enters as a premise rather than a measurement. To be precise about scope: the authors never claim a dose-response, they never state that a *larger* gap yields more meaning. Our study asks the questions the framing leaves open: through which channel does the objective act, does moving the gap move downstream usefulness, which of the probing results transfer to held-out sequences, and what determines when the method helps. The answers turn out to fit together into one mechanism.

## 2. Setup: architecture, objective, and metrics

Everything in this post uses the paper's Pythia-160m configuration: residual-stream activations $x\_t \in \mathbb{R}^d$ from layer 8, an SAE with $m = 16{,}384$ features, and $k = 20$ active features per token. The SAE is the standard encoder–decoder pair:

$$
f(x\_t) = \sigma(W\_\mathrm{enc} x\_t + b\_\mathrm{enc}), \qquad \hat{x}\_t = W\_\mathrm{dec}\, f(x\_t) + b\_\mathrm{dec}
\tag{1}
$$

where $\sigma$ is the **BatchTopK** activation: keep the largest activations across the batch, averaging $k = 20$ per token, and zero the rest. Sparsity is enforced by this top-k competition, a detail that becomes important in §8.

The dictionary is split into a **high-level group** (the first 20%, $h = 3{,}276$ features) and a **low-level group** (the remaining 13,108). Following the **Matryoshka** SAE design, the high-level group must also reconstruct the input *on its own*, which pushes it toward the largest, most global directions:

$$
L\_\mathrm{matr}(x\_t) = \lVert x\_t - \hat{x}\_t^{(H)} \rVert\_2^2 + \lVert x\_t - \hat{x}\_t \rVert\_2^2, \qquad \hat{x}\_t^{(H)} = W\_\mathrm{dec}^{0:h} f\_{0:h}(x\_t) + b\_\mathrm{dec}
\tag{2}
$$

Reconstruction quality is the **fraction of variance explained**, reported for the full dictionary (FVE) and for the high-level group alone ($\mathrm{FVE}\_\mathrm{high}$, computed from $\hat{x}^{(H)}$):

$$
\mathrm{FVE} = 1 - \frac{\mathrm{Var}(x - \hat{x})}{\mathrm{Var}(x)}
\tag{3}
$$

$\mathrm{FVE}\_\mathrm{high}$ tracks whether the high-level group still summarizes the input usefully; a group can look very smooth simply by reconstructing very little, so we report the two together throughout.

The temporal part is one added loss term, applied only to the high-level activations $z\_t = f\_{0:h}(x\_t)$. It is an **InfoNCE contrastive loss**: for each sequence in the batch, the high-level vector at token $t$ should be similar to the same sequence's vector at token $t-1$ (the "positive"), and dissimilar from the other sequences in the batch (the "negatives"):

$$
L = \sum\_{i=1}^{N} L\_\mathrm{matr}(x\_t^{(i)}) + \alpha\, L\_\mathrm{contr}
$$

$$
L\_\mathrm{contr} = -\frac{1}{N} \sum\_{i=1}^{N} \log \frac{\exp\!\big(s(z\_t^{(i)}, z\_{t-1}^{(i)})/\tau\big)}{\sum\_{j=1}^{N} \exp\!\big(s(z\_t^{(i)}, z\_{t-1}^{(j)})/\tau\big)} + (\text{symmetric term})
\tag{4}
$$

$N$ is the batch size (activations are loaded as shuffled $(x\_t, x\_{t-1})$ pairs), and the symmetric term repeats the expression with the roles of $t$ and $t-1$ exchanged. In the released implementation the similarity is the **raw inner product**, $s(u, v) = \langle u, v \rangle$, with temperature $\tau = 1$. A raw inner product is sensitive to both the *direction* of the feature vector and its overall *size* (norm), a property §4 examines. The negatives are what prevent the trivial solution where every high-level feature just goes constant. We call the loss weight `temp_alpha` ($\alpha = 1$ is the paper's setting).

Smoothness is the paper's Lipschitz-style metric: over a sequence, for each active feature take the largest per-token jump, normalized by how much the underlying activation moved, then average over active features and over sequences:

$$
\Delta\_s = \frac{1}{n'} \sum\_{i=1}^{n'} \max\_t \frac{\lvert f\_i(x\_t) - f\_i(x\_{t-1}) \rvert}{\lVert x\_t - x\_{t-1} \rVert\_2}, \qquad S = \mathrm{mean}\_\mathrm{seq}\, \Delta\_s
\tag{5}
$$

Computed separately for the two groups, this gives $S\_\mathrm{high}$ and $S\_\mathrm{low}$ (lower = steadier). The **smoothness gap** is $\mathrm{Gap} = S\_\mathrm{low} - S\_\mathrm{high}$; positive means the high-level group is the steadier one, which is the paper's disentanglement signal.

Finally, the downstream measurements. **Probes** are linear classifiers trained on frozen SAE features: *semantics* (the sequence's topic or category), *syntax* (the current token's part of speech), and *context* (which sequence a token came from, the paper's "question number" label). One distinction does a lot of work below: a probe evaluated on a **token-split** shares sequences between its train and test sets, so it can score well by recognizing a particular sequence; a **sequence-disjoint split** holds whole sequences out, so only structure that transfers to unseen sequences can score. The gap between the two splits measures how much of a probe's accuracy is sequence recognition.

## 3. Replication: the effect is real and tight

The paper's intrinsic results replicate on its released code, essentially exactly.

**Table 1.** Replication of the paper's headline intrinsic metrics.

| Metric | Paper | Our run |
| :--- | :---: | :---: |
| FVE (reconstruction) | 0.94 | 0.940 |
| Cosine similarity | 0.93 | 0.932 |
| Fraction of features alive | 0.87 | 0.936 |
| Smoothness, high group ($S\_\mathrm{high}$) | 0.09 | 0.087 |
| Smoothness, low group ($S\_\mathrm{low}$) | 0.17 | 0.163 |
| **Gap ($S\_\mathrm{low} - S\_\mathrm{high}$)** | **+0.08** | **+0.076** |

*The smoothness gap lands at +0.076 against the paper's +0.08 and is seed-tight: ±0.003 across re-training seeds. Reconstruction is preserved, as claimed.*

This deserves emphasis, because replications at this fidelity are not the norm: every number the paper reports for this setting checks out, the effect is robust to re-training, and the released code runs as documented. Question (A) is settled affirmatively. The rest of the study characterizes what the effect consists of and what it delivers.

## 4. Which channel of the features carries the smoothness gap?

A feature vector can change between tokens in two ways: its *direction* (which features are active, in what proportions) and its *magnitude* (how strongly they fire). The raw inner product in Eq. 4 responds to both. Our first set of experiments varied the similarity function and the loss form to localize which channel the measured gap lives in. One variant is a one-line change to the released loss, replacing the similarity with its scale-free counterpart:

$$
s(u, v) = \langle u, v \rangle \quad \longrightarrow \quad s(u, v) = \frac{\langle u, v \rangle}{\lVert u \rVert\_2\, \lVert v \rVert\_2}
\tag{6}
$$

Cosine similarity keeps the direction information and removes the magnitude channel. Everything else, architecture, data, training length, is unchanged.

A second variant strengthens the objective in a way that *keeps* the raw inner product: **SupCon** (supervised contrastive) enlarges the positive set from the single previous token to a causal window of the $W$ preceding tokens,

$$
L\_\mathrm{SupCon} = -\frac{1}{N} \sum\_{i=1}^{N} \frac{1}{\lvert P \rvert} \sum\_{p \in P} \log \frac{\exp\!\big(\langle z\_t^{(i)}, z\_{t-p}^{(i)} \rangle\big)}{\sum\_{j=1}^{N} \exp\!\big(\langle z\_t^{(i)}, z\_{t-p}^{(j)} \rangle\big)}, \qquad P = \{1, \dots, W\}
\tag{7}
$$

With $W = 1$ this reduces exactly to the paper's loss. We also tested two secondary variants: the paper's own Appendix-A *linearized* objective, which replaces the softmax with $L\_\mathrm{lin} = -\frac{1}{N}\sum\_i s\_{ii} + \frac{1}{N^2}\sum\_{i,j}(1 + s\_{ij})$, and a *Gram* (decoder-space) form that applies the contrastive term to the high-level reconstruction $W\_\mathrm{dec}^{0:h} z\_t$ instead of to $z\_t$.

**Table 2.** Loss variants: the gap tracks the magnitude channel of the similarity.

| Variant | Similarity $s(u,v)$ | Gap | $\mathrm{FVE}\_\mathrm{high}$ | Reading |
| :--- | :--- | :---: | :---: | :--- |
| **InfoNCE (released)** | $\langle u,v \rangle$ (raw) | +0.076 | 0.869 | reference run |
| InfoNCE, cosine | $\langle u,v \rangle / \lVert u \rVert \lVert v \rVert$ | −0.029 | 0.921 | gap inverts; reconstruction improves |
| InfoNCE, cosine, $\tau$=0.1 | cosine, sharper softmax | −0.019 | 0.920 | temperature does not restore the gap |
| Linearized (Appendix A) | normalized | −0.026 | — | gap appears only alongside dead features |
| Gram (decoder space) | raw, on $W\_\mathrm{dec} z$ | +0.030 | 0.18 | gap coincides with $\mathrm{FVE}\_\mathrm{high}$ collapse |
| Gram, cosine | cosine, decoder space | −0.007…−0.021 | 0.92 | $\mathrm{FVE}\_\mathrm{high}$ restored, gap gone |
| SupCon, $W$=5 | $\langle u,v \rangle$ (raw), window positives | +0.075 | 0.850 | gap preserved, the effect is robust |

*Cosine rows are seed-replicated (two seeds agree to 0.002); SupCon holds across two seeds and $W = 10$ (+0.078 both). The linearized objective can reach a large gap (+0.101) only at a strength that leaves ~23% of the dictionary permanently dead, a dead feature never fires and therefore reads as perfectly smooth, which is why we report all gaps at convergence together with the fraction of alive features.*

The pattern across the table is consistent: every manipulation that removes the magnitude channel from the similarity also removes (or inverts) the measured gap, and simultaneously *improves* $\mathrm{FVE}\_\mathrm{high}$ (0.87 → 0.92); the one manipulation that keeps the magnitude channel keeps the gap. So the smoothness gap, as measured by Eq. 5, is carried by the magnitude channel of the high-level features. Interestingly, the cosine model is a slightly better autoencoder that simply does not exhibit the gap, which suggested the gap and reconstruction quality are separate axes, and set up the next question.

SupCon also gave us the first downstream data point. It preserves the gap intact, and its probing outcome (10 matched splits) is neutral-to-slightly-negative: semantics −0.028, syntax −0.001, context −0.084 relative to the paper baseline. A variant can keep the smoothness gap while changing nothing downstream, the first indication that (A) and (B) vary independently.

## 5. Varying the contrastive weight: smoothness and downstream readability move independently

The direct test of the smoothness–usefulness link is to treat `temp_alpha` as a dial: sweep $\alpha \in \{0, 0.3, 1.0, 3.0, 10\}$, where $\alpha = 0$ is a matched vanilla Matryoshka SAE, identical in every other respect, and measure the gap and the three probes at every point.

**Table 3.** The contrastive-weight sweep. Gap, three probes (10 matched splits), and high-group reconstruction.

| temp_alpha | Gap | Semantics | Syntax | Context | $\mathrm{FVE}\_\mathrm{high}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 0 (vanilla) | −0.017 | 0.358 | 0.862 | 0.636 | 0.923 |
| 0.3 | +0.067 | 0.381 | 0.840 | 0.825 | 0.900 |
| **1.0 (paper)** | **+0.076** | **0.393** | **0.818** | **0.871** | **0.869** |
| 3.0 | +0.080 | 0.376 | 0.799 | 0.846 | 0.781 |
| 10.0 | +0.072 | 0.403 | 0.779 | 0.834 | 0.401 |

<figure class="tsae-fig">
<div class="chart-box">
<svg class="chart" viewBox="0 0 640 560" role="img" aria-label="Three aligned panels showing smoothness gap, FVE high, and probe accuracies across the contrastive weight sweep">
<!-- Panel A: gap -->
<text class="ptitle" x="56" y="20">A · Smoothness gap (S_low − S_high)</text>
<line class="grid" x1="56" y1="50.9" x2="530" y2="50.9"/>
<line class="grid" x1="56" y1="84.8" x2="530" y2="84.8"/>
<line class="zline" x1="56" y1="118.6" x2="530" y2="118.6"/>
<text x="50" y="54.4" text-anchor="end">0.08</text>
<text x="50" y="88.3" text-anchor="end">0.04</text>
<text x="50" y="122.1" text-anchor="end">0</text>
<polyline class="l1" points="56,133 174.5,61.9 293,54.3 411.5,50.9 530,57.7"/>
<circle class="p1" cx="56" cy="133" r="4"/><circle class="p1" cx="174.5" cy="61.9" r="4"/><circle class="p1" cx="293" cy="54.3" r="4"/><circle class="p1" cx="411.5" cy="50.9" r="4"/><circle class="p1" cx="530" cy="57.7" r="4"/>
<text class="dlabel" x="540" y="61">gap</text>
<!-- Panel B: FVE_high -->
<text class="ptitle" x="56" y="180">B · FVE_high (reconstruction from the high group alone)</text>
<line class="grid" x1="56" y1="221.5" x2="530" y2="221.5"/>
<line class="grid" x1="56" y1="258.2" x2="530" y2="258.2"/>
<line class="grid" x1="56" y1="294.8" x2="530" y2="294.8"/>
<text x="50" y="225" text-anchor="end">0.8</text>
<text x="50" y="261.7" text-anchor="end">0.6</text>
<text x="50" y="298.3" text-anchor="end">0.4</text>
<polyline class="ln" points="56,199 174.5,203.2 293,208.9 411.5,225 530,294.7"/>
<circle class="pn" cx="56" cy="199" r="4"/><circle class="pn" cx="174.5" cy="203.2" r="4"/><circle class="pn" cx="293" cy="208.9" r="4"/><circle class="pn" cx="411.5" cy="225" r="4"/><circle class="pn" cx="530" cy="294.7" r="4"/>
<text class="dlabel" x="540" y="297">FVE_high</text>
<!-- Panel C: probes -->
<text class="ptitle" x="56" y="340">C · Probe accuracy (10 matched splits)</text>
<line class="grid" x1="56" y1="354" x2="530" y2="354"/>
<line class="grid" x1="56" y1="404" x2="530" y2="404"/>
<line class="grid" x1="56" y1="454" x2="530" y2="454"/>
<line class="axis" x1="56" y1="504" x2="530" y2="504"/>
<text x="50" y="357.5" text-anchor="end">0.9</text>
<text x="50" y="407.5" text-anchor="end">0.7</text>
<text x="50" y="457.5" text-anchor="end">0.5</text>
<text x="50" y="507.5" text-anchor="end">0.3</text>
<polyline class="l2" points="56,363.5 174.5,369 293,374.5 411.5,379.3 530,384.3"/>
<polyline class="l3" points="56,420 174.5,372.8 293,361.3 411.5,367.5 530,370.5"/>
<polyline class="l1" points="56,489.5 174.5,483.8 293,480.8 411.5,485 530,478.3"/>
<circle class="p2" cx="56" cy="363.5" r="4"/><circle class="p2" cx="174.5" cy="369" r="4"/><circle class="p2" cx="293" cy="374.5" r="4"/><circle class="p2" cx="411.5" cy="379.3" r="4"/><circle class="p2" cx="530" cy="384.3" r="4"/>
<circle class="p3" cx="56" cy="420" r="4"/><circle class="p3" cx="174.5" cy="372.8" r="4"/><circle class="p3" cx="293" cy="361.3" r="4"/><circle class="p3" cx="411.5" cy="367.5" r="4"/><circle class="p3" cx="530" cy="370.5" r="4"/>
<circle class="p1" cx="56" cy="489.5" r="4"/><circle class="p1" cx="174.5" cy="483.8" r="4"/><circle class="p1" cx="293" cy="480.8" r="4"/><circle class="p1" cx="411.5" cy="485" r="4"/><circle class="p1" cx="530" cy="478.3" r="4"/>
<text class="dlabel" x="540" y="374">context</text>
<text class="dlabel" x="540" y="389">syntax</text>
<text class="dlabel" x="540" y="482">semantics</text>
<!-- shared x axis labels -->
<text x="56" y="524" text-anchor="middle">0</text>
<text x="56" y="537" text-anchor="middle" font-size="9.5">(vanilla)</text>
<text x="174.5" y="524" text-anchor="middle">0.3</text>
<text x="293" y="524" text-anchor="middle">1.0</text>
<text x="293" y="537" text-anchor="middle" font-size="9.5">(paper)</text>
<text x="411.5" y="524" text-anchor="middle">3.0</text>
<text x="530" y="524" text-anchor="middle">10</text>
<text x="293" y="555" text-anchor="middle">contrastive weight temp_alpha (ordinal spacing)</text>
</svg>
</div>
<div class="legend" aria-hidden="true">
<span><span class="sw s1"></span>semantics / gap</span>
<span><span class="sw s2"></span>syntax</span>
<span><span class="sw s3"></span>context</span>
<span><span class="sw sn"></span>FVE_high</span>
</div>
</figure>

*Figure 1. Panel A: the gap is created by the contrastive term (vanilla ≈ 0), is ~90% of maximum by $\alpha = 0.3$, and peaks at +0.080 near $\alpha = 3$. Panel B: $\mathrm{FVE}\_\mathrm{high}$ declines past $\alpha \approx 1$ and drops steeply by $\alpha = 10$, bounding the usable range to roughly $\alpha \in [0.3, 1]$. Panel C: while the gap traverses its full range, semantic accuracy stays inside a flat 0.36–0.40 band, syntax declines monotonically, and context, the probe most directly aligned with what the loss optimizes, is the only one that rises.*

The three probe trajectories decompose cleanly. **Semantics does not track the gap**, the central empirical fact of the study, and the direct answer to question (B) in this regime: making the features smoother, by itself, does not make them more semantically readable. **Syntax declines steadily** (0.862 → 0.779), which is expected from the design, encouraging adjacent high-level vectors to agree averages out token-level information, but worth quantifying as the price paid. **Context rises with the gap**, and here the interpretation needs care: the contrastive term pulls same-sequence tokens together, and the context probe asks which sequence a token belongs to, so this probe improves because it measures a quantity the loss directly optimizes. §6 tests how much of it transfers to unseen sequences.

Two controls sharpen the interpretation. First, a **magnitude-only objective** that matches just the norms of adjacent high-level vectors, leaving direction free:

$$
L\_\mathrm{mag} = \frac{1}{N} \sum\_{i=1}^{N} \Big( \lVert z\_t^{(i)} \rVert\_2 - \lVert z\_{t-1}^{(i)} \rVert\_2 \Big)^2
\tag{8}
$$

This produces *no* gap (+0.007 at gentle weight; −0.018 when pushed hard enough to deactivate 30% of features). Together with §4, this pins down the relationship precisely: the gap is *measured through* the magnitude channel but is *not produced by* magnitude matching alone, it is a genuine consequence of the full contrastive constraint. Second, a per-feature whitening diagnostic (normalizing each feature by its temporal variance, the axis a slow-feature-analysis approach would standardize) turns out to manufacture a +0.126 gap on the vanilla model with no temporal training at all, ultra-sparse features divided by tiny variances read as jumpy. We flag this because it means the seemingly safer "unit-variance" version of the smoothness metric has its own artifact mode; the matched vanilla baseline is what revealed it.

## 6. The probing evidence under sequence-disjoint splits

The sweep used our probe suite. The stronger test is to rebuild the paper's own probing protocol, encode the last ~25 tokens of each sequence; probe for category (semantics) and sequence identity (context) from the high-level features, and evaluate it both ways: the paper's token-split, and a sequence-disjoint split.

One methodological result first, because it shaped everything after it. Concerned that "topic" was too coarse an instrument, we built what we considered a fairer, genuinely slow semantic probe: **entity-in-focus**, which named entity (person, organization, place) is currently being discussed at each token. Seed-replicated (n = 3 per arm), it produced the first result in the study where a downstream metric tracked the contrastive weight: +0.032 for the temporal model, Welch t ≈ 12. When we re-evaluated it with document-disjoint splits, the difference was −0.005, zero. The probe had been reading *document identity* (each document has a characteristic mix of entities) rather than transferable entity information. We adopted sequence- or document-disjoint evaluation for every measurement after this one; this example is why, and we suspect the same check would be informative for other probing pipelines in the literature.

**Table 4.** Paper-protocol probing, temporal vs. matched vanilla, token-split → sequence-disjoint.

| Topic set | Category distinctness | Semantic advantage (token-split → seq-disjoint) | Context advantage (seq-disjoint) | Syntax |
| :--- | :---: | :---: | :---: | :---: |
| MMLU distinct-9 | high | +0.104 → +0.087 | 0.000 | −0.17 |
| MMLU distinct-4 | high | +0.070 → +0.089 | 0.000 | −0.16 |
| MMLU subtle-4 (sciences) | low | +0.065 → +0.050 | 0.000 | −0.13 |
| FineFineWeb (10 web domains) | fuzzy | +0.044 → −0.010 | 0.000 | −0.18 |

*Token-split context advantages of +0.245 to +0.375 reduce to exactly 0.000 on every dataset under sequence-disjoint evaluation, for both models. The semantic advantage survives the control on all MMLU variants and does not on FineFineWeb web text.*

Two findings, pointing in different directions:

**The context probe measures sequence recognition.** Under sequence-disjoint splits its advantage is exactly zero everywhere, a sequence identity cannot be predicted for sequences the probe never saw, for either model. This is the expected behavior given the objective: the contrastive term groups tokens by sequence, and the token-split context probe scores exactly that grouping. So we read the paper's context result as a measurement of the training objective itself rather than of transferable contextual information, and our main suggestion to the authors (made in our feedback to them) is to report this probe under a sequence-disjoint split, a single extra row that would settle it in their own setting.

**The semantic advantage is genuine on the paper's own data.** On MMLU, short, curated, single-topic exam text, the temporal advantage survives held-out-sequence evaluation at +0.05 to +0.09. The token-split inflates it only modestly (~15–20%, e.g. +0.104 → +0.087); the effect itself is real. On FineFineWeb's messier multi-topic web text, the advantage does not appear (−0.010). This is the boundary condition that Section 7 explains. Syntax behaves as a consistent control throughout: the temporal model is 0.13–0.18 lower, matching the paper's design intent (syntax routed to the low-level group) and the cost measured in §5.

Worth recording: an earlier, FineFineWeb-only version of this experiment had led us to a stronger negative reading. Running the paper's own dataset corrected us, the cross-dataset design is what produced the accurate, two-sided picture, and it is the reason we regard the MMLU semantic result as one of the paper's claims that our study *strengthens*: it now stands with the leakage removed.

## 7. A sharpening mechanism: the benefit tracks the base model's existing structure

Why does the semantic benefit appear on MMLU and not on FineFineWeb? A zero-training diagnostic on the raw layer-8 activations answers this by eliminating one candidate and identifying another.

The candidate it eliminates is the contrastive positive. If web text simply made "previous token" a bad positive (say, because sentences change topic more), that would explain the difference, but the measurement says otherwise: the previous token is an equally good positive on both datasets (mean-centered cosine margin over a random cross-sequence negative: +0.309 on MMLU, +0.332 on FineFineWeb; ~97% of positives fall in the same sentence on both). The candidate it identifies is the **base model's category geometry**: within-category minus between-category cosine of sequence centroids, in the matched baseline SAE's high-level space, is **+0.330 on MMLU versus +0.038 on FineFineWeb**, roughly an 8× difference in how much the underlying representation already separates the categories.

This suggests a simple account: the contrastive loss reinforces sequence-level clustering, which converts category structure *already present* in the representation into a more linearly readable form. Where the base model separates categories (MMLU), there is structure to sharpen; where it barely does (FineFineWeb), there is not. **The objective sharpens existing structure rather than creating new structure.** This account also retroactively explains every result in §4–5 where changing the objective left semantic accuracy unchanged: the ceiling was never set by the objective.

The account makes a quantitative prediction. Define $\mathrm{sep} = \eta^2$, the fraction of the raw activations' variance that lies between categories (a standard effect-size measure of linear separability), and $\mathrm{coh}$ = within-sequence coherence, how similar tokens of the same sequence are to one another in the raw representation. Then across labeled corpora the semantic gain of temporal over vanilla should scale with their product:

$$
\mathrm{gain} \approx 9.63 \cdot (\mathrm{sep} \times \mathrm{coh}) - 0.022, \qquad r = 0.89
\tag{9}
$$

<figure class="tsae-fig">
<div class="chart-box">
<svg class="chart" viewBox="0 0 640 330" role="img" aria-label="Scatter plot of semantic gain versus base-model separability times coherence, four corpora, with fitted line">
<line class="grid" x1="64" y1="24" x2="620" y2="24"/>
<line class="grid" x1="64" y1="68.7" x2="620" y2="68.7"/>
<line class="grid" x1="64" y1="113.4" x2="620" y2="113.4"/>
<line class="grid" x1="64" y1="158.2" x2="620" y2="158.2"/>
<line class="zline" x1="64" y1="202.9" x2="620" y2="202.9"/>
<line class="grid" x1="64" y1="247.6" x2="620" y2="247.6"/>
<line class="axis" x1="64" y1="270" x2="620" y2="270"/>
<text x="58" y="27.5" text-anchor="end">0.08</text>
<text x="58" y="72.2" text-anchor="end">0.06</text>
<text x="58" y="116.9" text-anchor="end">0.04</text>
<text x="58" y="161.7" text-anchor="end">0.02</text>
<text x="58" y="206.4" text-anchor="end">0</text>
<text x="58" y="251.1" text-anchor="end">−0.02</text>
<text x="64" y="288" text-anchor="middle">0</text>
<text x="203" y="288" text-anchor="middle">0.002</text>
<text x="342" y="288" text-anchor="middle">0.004</text>
<text x="481" y="288" text-anchor="middle">0.006</text>
<text x="620" y="288" text-anchor="middle">0.008</text>
<line class="refline" x1="64" y1="252.1" x2="620" y2="79.8"/>
<text x="84" y="52">fit: gain ≈ 9.63·(sep×coh) − 0.022</text>
<circle class="p1" cx="140.5" cy="214.1" r="5"/>
<circle class="p1" cx="307.3" cy="194" r="5"/>
<circle class="p1" cx="529.7" cy="144.8" r="5"/>
<circle class="p1" cx="578.3" cy="55.3" r="5"/>
<text class="vlabel" x="150" y="232">MMLU subtle-4</text>
<text class="vlabel" x="317" y="212">FineFineWeb</text>
<text class="vlabel" x="517" y="148" text-anchor="end">MMLU distinct-4</text>
<text class="vlabel" x="566" y="59" text-anchor="end">MMLU distinct-9</text>
<text x="342" y="316" text-anchor="middle">sep × coh (base-model separability × within-sequence coherence)</text>
</svg>
</div>
</figure>

**Figure 2.** Semantic gain (temporal − vanilla, sequence-disjoint) against $\mathrm{sep} \times \mathrm{coh}$ for the four corpora:

| Corpus | $\mathrm{sep} \times \mathrm{coh}$ | Semantic gain |
| :--- | :---: | :---: |
| MMLU subtle-4 | 0.0011 | −0.005 |
| FineFineWeb | 0.0035 | +0.004 |
| MMLU distinct-4 | 0.0067 | +0.026 |
| MMLU distinct-9 | 0.0074 | +0.066 |

*Caveat, stated plainly: two of the four corpora are MMLU-derived siblings, so this is effectively N ≈ 3, a consistent trend that supports the mechanism, not an established law. §11 describes the out-of-sample test that would upgrade it.*

## 8. Testing whether the objective adds new capabilities

The sharpening account makes a strong prediction: experiments designed to have the objective *create* structure the base representation lacks should come back at the base model's ceiling. We ran four such tests, from four different directions. All four results are consistent with the sharpening account, and one of them additionally reveals the mechanism by which strong contrastive weights move information out of the slow group.

**Table 5.** Four capability tests. "Raw ceiling" = a linear probe on the raw residual-stream activations, the most the underlying representation makes linearly available. "Diff-in-means" = the difference between class-mean activations used as a steering direction, a one-line supervised baseline.

| Experiment | Question | Key numbers | Reading |
| :--- | :--- | :--- | :--- |
| Oracle-label training (FineFineWeb) | Can perfect category labels fed into training raise topic readability past the raw activations? | raw ceiling 0.318; best label-assisted setting 0.319 vs. plain 0.307 (within ±0.05 noise); 0.167 at full strength | at every strength that preserves reconstruction, supervision ties the ceiling; stronger pulls degrade the model |
| Dialogue state (Schema-Guided Dialogue) | Does the temporal model track conversational state, the natural use case for slow features? | overall goal +0.117 over raw, decomposing into topic +0.055 and within-topic state −0.022 (raw already 0.891) | the gain is topic sharpening; per-turn state is near-saturated in the base model |
| Belief-state world (HMM, exact labels) | Can the slow group recover exact latent beliefs a vanilla SAE also sees? | vanilla R² 0.936; any $\alpha \geq 0.03$ gives ≤ 0.127; full dictionary pinned at ≈ 0.598 throughout | dose-dependent; content moves out of the slow group, not out of the SAE |
| Steering (math / law domains) | Are temporal features better residual-stream steering directions? | prompting ≈ diff-in-means > SAE features; temporal ≈ vanilla (best vanilla feature 0.77 vs. temporal 0.45 in math) | no steering advantage from the objective under our protocol (which differs from the paper's, §11) |

The dialogue experiment deserves one more sentence, because at first pass it looked like the method's best result anywhere: +0.117 over raw on predicting the user's current goal, transferring cleanly to held-out conversations. The decomposition is what makes it interpretable: the 23 goals are mostly tied to the *service* (restaurants vs. buses), so a topic-sharpener scores well on them without tracking any conversational state. Holding the topic fixed and asking only about the stage of the conversation (searching vs. booking) removes the entire advantage. This is the sharpening account appearing in a new domain, with a larger apparent effect because conversations stay unusually on-topic, exactly the high-coherence regime Eq. 9 predicts the method likes.

### The belief-state world: a dose-response with a visible mechanism

The most controlled of the four tests uses a synthetic hidden-state world (a sticky HMM) where the exact Bayesian belief over hidden states is available in closed form. A small transformer learns the world; SAEs are trained on its residual stream; we measure how much of the true belief simplex the slow group alone linearly recovers (ridge R²), sweeping both the temporal weight ($\alpha \in \{0, 0.01, 0.03, 0.1, 0.3, 1.0\}$) and, to check whether any single setting was at fault, the learning rate, with 5 seeds per cell.

**Table 6.** The belief-state dose-response.

| Setting | Slow-group belief R² | Slow-group mean activation |
| :--- | :---: | :---: |
| vanilla | 0.936 ± 0.022 | ~3.5 |
| $\alpha$ = 0 (loss off) | 0.846 ± 0.179 | ~3.4 |
| $\alpha$ = 0.01 | 0.720 ± 0.225 | 1.7 |
| $\alpha$ = 0.03 | 0.127 ± 0.163 | 1.2 |
| $\alpha$ = 0.1 | 0.054 ± 0.017 | 1.0 |
| $\alpha$ = 1.0 | 0.044 ± 0.004 | 1.3 |

<figure class="tsae-fig">
<div class="chart-box">
<svg class="chart" viewBox="0 0 640 400" role="img" aria-label="Two aligned panels: belief-state R-squared from the slow group and slow-group mean activation, both against the temporal loss weight">
<!-- Panel A: R2 -->
<text class="ptitle" x="56" y="20">A · Belief R² from the slow group alone (dwell 0.5, 5 seeds, ±1 sd)</text>
<line class="grid" x1="56" y1="34" x2="530" y2="34"/>
<line class="grid" x1="56" y1="74" x2="530" y2="74"/>
<line class="grid" x1="56" y1="114" x2="530" y2="114"/>
<line class="grid" x1="56" y1="154" x2="530" y2="154"/>
<line class="axis" x1="56" y1="194" x2="530" y2="194"/>
<text x="50" y="37.5" text-anchor="end">1.0</text>
<text x="50" y="117.5" text-anchor="end">0.5</text>
<text x="50" y="197.5" text-anchor="end">0</text>
<line class="refline" x1="56" y1="44.2" x2="530" y2="44.2"/>
<text class="vlabel" x="540" y="47">vanilla</text>
<text x="540" y="60" font-size="10">R² 0.936</text>
<line class="err" x1="56" y1="34" x2="56" y2="87.3"/>
<line class="err" x1="174.5" y1="42.8" x2="174.5" y2="114.8"/>
<line class="err" x1="293" y1="147.6" x2="293" y2="194"/>
<line class="err" x1="411.5" y1="182.6" x2="411.5" y2="188.1"/>
<line class="err" x1="530" y1="186.3" x2="530" y2="187.6"/>
<polyline class="l1" points="56,58.6 174.5,78.8 293,173.7 411.5,185.4 530,187"/>
<circle class="p1" cx="56" cy="58.6" r="4"/><circle class="p1" cx="174.5" cy="78.8" r="4"/><circle class="p1" cx="293" cy="173.7" r="4"/><circle class="p1" cx="411.5" cy="185.4" r="4"/><circle class="p1" cx="530" cy="187" r="4"/>
<!-- Panel B: mean act -->
<text class="ptitle" x="56" y="230">B · Mean activation magnitude of the slow group</text>
<line class="grid" x1="56" y1="244" x2="530" y2="244"/>
<line class="grid" x1="56" y1="299" x2="530" y2="299"/>
<line class="axis" x1="56" y1="354" x2="530" y2="354"/>
<text x="50" y="247.5" text-anchor="end">4</text>
<text x="50" y="302.5" text-anchor="end">2</text>
<text x="50" y="357.5" text-anchor="end">0</text>
<line class="refline" x1="56" y1="257.8" x2="530" y2="257.8"/>
<text class="vlabel" x="540" y="261">vanilla</text>
<text x="540" y="274" font-size="10">≈ 3.5</text>
<polyline class="l2" points="56,260.5 174.5,307.3 293,321 411.5,326.5 530,318.3"/>
<circle class="p2" cx="56" cy="260.5" r="4"/><circle class="p2" cx="174.5" cy="307.3" r="4"/><circle class="p2" cx="293" cy="321" r="4"/><circle class="p2" cx="411.5" cy="326.5" r="4"/><circle class="p2" cx="530" cy="318.3" r="4"/>
<!-- shared x labels -->
<text x="56" y="374" text-anchor="middle">0 (off)</text>
<text x="174.5" y="374" text-anchor="middle">0.01</text>
<text x="293" y="374" text-anchor="middle">0.03</text>
<text x="411.5" y="374" text-anchor="middle">0.1</text>
<text x="530" y="374" text-anchor="middle">1.0</text>
<text x="293" y="394" text-anchor="middle">temporal loss weight temp_alpha (ordinal spacing)</text>
</svg>
</div>
<div class="legend" aria-hidden="true">
<span><span class="sw s1"></span>slow-group belief R²</span>
<span><span class="sw s2"></span>slow-group mean activation</span>
<span><span class="sw sref"></span>vanilla SAE reference</span>
</div>
</figure>

*Figure 3. Belief content in the slow group falls with a threshold near $\alpha \approx 0.01$–0.03 and never returns; activation magnitude falls in lockstep. "0 (off)" is the temporal architecture with the contrastive term multiplied by zero, it matches the vanilla reference, as it should. The learning-rate sweep at $\alpha = 0.1$ (lr ∈ {1e-4, 3e-4, 1e-3}) gives R² of 0.108 / 0.056 / 0.069 at dwell 0.9, all far below vanilla's 0.444, so the pattern is not a learning-rate artifact. The full-dictionary R² stays pinned at ≈ 0.598 in every arm.*

The mechanism is visible in the two panels of Figure 3 read together. The contrastive term reduces the slow group's activation magnitude (≈3.5 → ≈1.0); under BatchTopK, features compete globally for the k active slots, so lower-magnitude slow features lose their slots, and the belief information relocates to the rest of the dictionary, which is why the full-dictionary R² never moves. The information is not destroyed; it migrates out of the group the objective is meant to enrich. There is a narrow regime ($\alpha \approx 0.01$) where the slow group retains its content, but there the temporal effect is correspondingly small: on this architecture, the objective's strength and the slow group's information content trade off on the same knob. This observation suggests its own fix, reserving top-k slots per group would decouple the objective from the slot competition, which we list in §11 as a constructive next experiment.

## 9. Characterizing the slowness itself

Finally, a set of zero-training measurements on the finished checkpoints asks what kind of slowness the objective actually produces. This section contains both the study's clearest support for the paper's core intuition and the sharpest boundary on it.

The support: measured per feature, with frequency-matched and shuffle nulls, temporal training **does** create genuinely more persistent features, this is not a magnitude illusion, and it corrected our own earlier lean toward a "it's all magnitude" reading.

**Genuine persistence, null-corrected.**

| Model | Genuine persistence (null-corrected) | Autocorr. boost vs vanilla (lag 1 / 2 / 4) | Mean active magnitude |
| :--- | :---: | :---: | :---: |
| Temporal (paper) | +0.172 | +0.044 / +0.012 / 0.000 | 2.09 |
| Vanilla ($\alpha$ = 0, 3 seeds) | +0.126 ± 0.001 | — | 2.97 |

<figure class="tsae-fig">
<div class="chart-box">
<svg class="chart" viewBox="0 0 640 250" role="img" aria-label="Bar chart of support-autocorrelation boost over vanilla at lags one, two, and four">
<text class="ptitle" x="56" y="20">Support-autocorrelation boost over vanilla, by lag (high group)</text>
<line class="grid" x1="56" y1="62" x2="610" y2="62"/>
<line class="grid" x1="56" y1="126" x2="610" y2="126"/>
<line class="axis" x1="56" y1="190" x2="610" y2="190"/>
<text x="50" y="65.5" text-anchor="end">0.04</text>
<text x="50" y="129.5" text-anchor="end">0.02</text>
<text x="50" y="193.5" text-anchor="end">0</text>
<path class="bar" d="M 112.3 190 L 112.3 53.2 Q 112.3 49.2 116.3 49.2 L 180.3 49.2 Q 184.3 49.2 184.3 53.2 L 184.3 190 Z"/>
<path class="bar" d="M 297 190 L 297 155.6 Q 297 151.6 301 151.6 L 365 151.6 Q 369 151.6 369 155.6 L 369 190 Z"/>
<line class="l1" x1="481.7" y1="190" x2="553.7" y2="190" stroke-width="3"/>
<text class="vlabel" x="148.3" y="40" text-anchor="middle">+0.044</text>
<text class="vlabel" x="333" y="142" text-anchor="middle">+0.012</text>
<text class="vlabel" x="517.7" y="180" text-anchor="middle">0.000</text>
<text x="148.3" y="212" text-anchor="middle">lag 1</text>
<text x="333" y="212" text-anchor="middle">lag 2</text>
<text x="517.7" y="212" text-anchor="middle">lag 4</text>
<text x="333" y="238" text-anchor="middle">token offset at which persistence is measured</text>
</svg>
</div>
</figure>

*Figure 4. The +0.046 excess is far outside the seed band (2σ = 0.002). It is not magnitude, the temporal model's persistent features fire at lower magnitude (2.09 vs 2.97), and not positional (mean |correlation with absolute position| ≈ 0.01; the slowest quartile contains no position-tracking features). The added persistence is concentrated at the exact offset the loss trains: the contrastive positive in Eq. 4 is the token at offset 1, and the persistence boost is +0.044 at lag 1, +0.012 at lag 2, and indistinguishable from vanilla by lag 4. The objective produces episode-lengthening ("once on, stay on for about one more token"), not long-range slowness.*

Three follow-up measurements locate the boundaries of this effect.

**Longer offsets do not buy longer timescales.** The natural generalization, a "timescale ladder" that contrasts different nested groups against positives at offsets 32, 8, and 1, turns out not to produce a multi-timescale decomposition. Only the offset-1 group responds to its objective; and a matched control (the same four-group Matryoshka, no temporal term at all) shows the apparent slow-to-fast ordering of the groups arises from Matryoshka nesting by itself, early nested groups grab dominant, slowly-varying directions because they must reconstruct alone. That is a useful free observation in its own right: *Matryoshka allocation alone provides a crude timescale prior*, with no temporal objective needed.

| Group (training offset) | Ladder (cosine / raw dot) | No-temporal control | Boost from contrastive term |
| :--- | :---: | :---: | :---: |
| g0 (offset 32) | 0.075 / 0.080 | 0.078 | ≈ 0 |
| g1 (offset 8) | 0.074 / 0.067 | 0.074 | ≈ 0 |
| g2 (offset 1) | 0.197 / 0.219 | 0.120 | +0.08 / +0.10 |

*Support autocorrelation at each group's own training lag. Reconstruction stays healthy in all arms. The per-token contrastive form appears to be a lag-1 tool regardless of the offset it is aimed at, consistent with the constraint that a per-token encoder has no state with which to express longer-range dependence.*

**A linear baseline is smoother at every lag.** A closed-form slow feature analysis (SFA), whiten the layer-8 activations, eigendecompose the covariance of their temporal differences, keep the slowest directions, yields a linear slow subspace with magnitude-autocorrelation 0.82 / 0.58 / 0.43 at lags 1 / 8 / 64, versus roughly 0.08 for *any* SAE feature, temporal or vanilla. Sparse thresholded features are simply a high-turnover representation; sparsity, not the temporal term, dominates the smoothness budget. On the positive side, the slowest high-group features in *both* models are clean, interpretable topic and register detectors (mathematical-proof register, legal text, astronomy, source-code headers), the paper's qualitative picture of what slow features look like is accurate; the temporal objective makes the same kind of feature moderately stickier.

**Two auxiliary properties, briefly.** On the authors' future-work suggestion of using slow-feature changes for change-point detection: with a matched control (synthetic document seams that change document but not domain), all signals we tested, raw activations, vanilla, temporal, respond to lexical novelty rather than topic change (cross-domain vs. same-domain selectivity ≈ 0 for every arm); within that limit, temporal-high is a genuinely better document-boundary detector than raw activations (average precision 0.060 vs. 0.030), and its signal is directional (cosine) rather than magnitude-based. And on cross-seed reproducibility, the two standard lenses disagree in an instructive way:

| Matching criterion | Vanilla | Temporal | Δ |
| :--- | :---: | :---: | :---: |
| Decoder cosine (field-standard) | 0.702 | 0.725 | +0.023 |
| Encoder cosine | 0.605 | 0.506 | −0.099 |
| Activation-pattern correlation | 0.637 | 0.560 | −0.076 |

*By decoder-direction similarity the temporal features look slightly more stable across seeds; by what the features actually do (encoder directions, activation patterns on shared tokens) they are somewhat less so. Measured at $\alpha = 0.3$; random-direction null max-cosine is 0.128.*

## 10. Summary: what the temporal objective does, and measurement lessons

Assembling the pieces into one description of the method:

> The temporal contrastive objective makes high-level SAE features genuinely more persistent, a real, seed-robust, non-magnitude effect, concentrated at the single token offset the loss trains. That persistence registers in the smoothness gap through the features' magnitude channel, but moving the gap does not move semantic readability. The method's reliable downstream benefit is sharpening: where the base model already separates categories and sequences stay coherent, the objective makes that structure more linearly readable (+0.05 to +0.09 on MMLU under held-out-sequence evaluation); where the base structure is weak, the benefit does not appear. The cost is a consistent reduction in token-level (syntax) readability, and at strong weights, migration of information out of the slow group via top-k competition.

On the paper's individual claims, our ledger is mixed in both directions. The smoothness gap: replicates exactly, robust across seeds and loss variants that preserve the magnitude channel. The MMLU semantic advantage: *strengthened*, it survives a leakage control the original evaluation did not include. The interpretability of slow features: confirmed qualitatively. The context advantage: does not transfer to held-out sequences on any dataset we tested, and we read it as the loss measuring itself; we have suggested the sequence-disjoint row to the authors as the single most informative addition. The implicit smoothness-causes-semantics mechanism: not supported, the two quantities move independently under the weight sweep.

The transferable lessons are mostly about measurement, and none of them is specific to this paper:

- **Optimization targets leak into evaluations that share their structure.** A contrastive loss that groups tokens by sequence will improve any probe whose label correlates with sequence identity, context probes, entity mixes, dialogue services. Sequence-disjoint evaluation separates this from transferable structure, and in our hands it did so every time it was applied.
- **Headline metrics can decouple from the quantity they stand in for.** The smoothness gap is not semantic quality; decoder-cosine similarity is not functional identity; token-split probe accuracy is not generalization. In each case the standard metric and a more direct measurement disagreed.
- **Matched controls (same architecture, objective off) are the highest-value experiment per GPU-hour.** They exposed the whitening artifact, the Matryoshka timescale ordering, and the vanilla-equivalence of the $\alpha = 0$ arm, and twice corrected our own preliminary readings, once in each direction.
- **Two-sided characterizations beat verdicts.** Our early FineFineWeb-only result would have supported an overly negative conclusion; the paper's MMLU-centric evaluation supported an overly general one. The cross-dataset design is what located the actual boundary.

## 11. Scope, limitations, and next experiments

Everything here is one model (Pythia-160m), one layer (8), one SAE configuration, and a specific probe suite. The paper also reports Gemma2-2b and Llama-3.1-8b, which we have not probed, and its headline probing figure is on Gemma2-2b. Our steering comparison used a simpler protocol than the paper's LLM-judged success-versus-coherence evaluation, so it is adjacent evidence rather than a test of that figure. Eq. 9 rests on effectively three independent corpora. And the belief-world SAE hyperparameters, while swept over loss weight and learning rate, were not swept architecturally.

The sharpening mechanism makes a falsifiable prediction we want to register before running it: on Gemma2-2b, (a) the context advantage should again reduce to ≈ 0 under sequence-disjoint splits, and (b) the sequence-disjoint semantic advantage should be *larger* than Pythia's +0.09 if and only if Gemma's $\mathrm{sep} \times \mathrm{coh}$ is larger, measurable from raw activations in half an hour, before any training. Outcome (b) failing would count against the sharpening account; the semantic advantage surviving on a corpus with low base separation would count strongly against it.

Three experiments top our list, each aimed at a specific open joint in the analysis:

- **The post-hoc smoothing null model.** Apply a causal exponential moving average, $\tilde{f}\_t = \lambda \tilde{f}\_{t-1} + (1-\lambda) f\_t$, to a *vanilla* SAE's high-level features at inference time, and run it through the entire evaluation stack. If this zero-parameter filter reproduces the temporal SAE's profile, the objective is amortized smoothing and the field gets it for free; if probes drop because filtering blurs token identity while trained temporal features do not, that is the first evidence the objective does something a filter cannot, which would genuinely strengthen the method's case. Either outcome is informative, and no one (paper included) has run it.
- **Reserved per-group top-k slots.** §8 traced the slow group's information loss to top-k competition, not necessarily to the objective itself. Guaranteeing the slow group a fixed share of the k slots and re-running the belief-world sweep separates the two: if belief R² survives at effective $\alpha$, the fix is architectural and the objective is rehabilitated for state-tracking; if not, the limitation is in the objective. This is the constructive experiment our mechanism finding most directly motivates.
- **The Gemma2-2b leak-control replication**, with the prediction above registered first, simultaneously the strongest test of the sharpening law and the removal of our study's main external-validity limitation.

All numbers are from the reports of a single reproduction-and-analysis campaign on Pythia-160m layer 8, run on the authors' released code with extensions as isolated subclasses. The findings concern the paper's claims in the paper's own regime and should not be read as a general statement about slowness priors. We have shared a four-point summary of these results with the authors.
