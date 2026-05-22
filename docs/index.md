# Welcome to Jeo

<div style="text-align: left">
<img align="right" src="images/jeo_logo_blue.png" width="100">
</div>

<div style="
  background-color: #01579b;
  padding: 12px 20px;
  text-align: center;
  color: white;
  font-weight: bold;
  font-size: 1.2em;
  border-radius: 8px;
  margin-bottom: 16px;
">
  Geospatial ML — from satellite imagery to trained models.
</div>

<div class="mdx-hero__content">
  <a href="https://github.com/google-deepmind/jeo" class="md-button">
    GitHub repository
  </a>
</div>

---

## What is Jeo?

Jeo is an open source framework developed by Google DeepMind for **machine
learning on geospatial remote sensing and Earth observation (EO) data**. It uses
[JAX](https://github.com/google/jax) and
[Flax](https://github.com/google/flax) for high-performance model training and
inference on large-scale geospatial datasets.

Jeo is primarily intended for researchers and developers working in geospatial
analysis, remote sensing, environmental science, and sustainability modelling.

## Where to Start

| Your Goal | Go To |
|-----------|-------|
| **🌍 New to Jeo?** Complete end-to-end tutorial | [Starter Tutorial](starter.md) |
| **⚙️ Understand configs** and how to write them | [Configs](configs.md) |
| **🚀 Launch an experiment** on XManager or locally | [Launching](launching.md) |
| **📦 Start a new project** from scratch | [New Project Guide](new_project.md) |
| **🔧 Troubleshooting** common errors | [FAQ](faq.md) |

## Documentation

### Getting Started

-   **[Starter Tutorial](starter.md)** — A hands-on, phase-by-phase walkthrough
    using a real dataset (GlobalGeoTree). Covers the full pipeline from data
    discovery through GeeFlow export, model training, and evaluation.

### Core Concepts

-   **[Configs](configs.md)** — Config system, custom modules, duration specs,
    sweeps, Vizier integration, and loading pretrained models.
-   **[Models](models.md)** — Model architectures, input/output conventions,
    weight loading, and module resolution.
-   **[Tasks & Losses](tasks.md)** — TaskBase interface, loss functions, and
    available task types.
-   **[Datasets](datasets.md)** — TFDS datasets, custom loaders, and GeeFlow
    integration.
-   **[Preprocessing](pp.md)** — The pipe-based preprocessing grammar and
    available operations.
-   **[Evaluators](evaluators.md)** — Evaluator interface, metrics, Lumascope
    visualization.

### Guides

### Reference

-   **[Conventions & Style](conventions.md)** — Coding conventions, abbreviations,
    config style guide, and design principles.
-   **[FAQ & Troubleshooting](faq.md)** — Common errors, memory limits, GPU OOM,
    GeeFlow authentication, and debugging tips.

## Design

Jeo's code structure is inspired by
[Big Vision](https://github.com/google-research/big_vision) and
[Scenic](https://github.com/google-research/scenic), and builds upon:

-   **[JAX](https://github.com/google/jax)** — High-performance computation
    engine.
-   **[Flax](https://github.com/google/flax)** — Neural network building tools.
-   **[`tf.data`](https://www.tensorflow.org/guide/data)** — Data input
    pipelines.
-   **[GeeFlow](https://github.com/google-deepmind/geeflow)** — Connects to
    [Google Earth Engine](https://earthengine.google.com/)'s satellite data
    resources for large-scale dataset generation.

By using JAX and Flax, Jeo benefits from automatic differentiation, XLA
compilation, and seamless execution across CPUs, GPUs, and
[Google Cloud TPUs](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms).
This focus on performance is particularly relevant for Earth observation tasks,
which often involve processing massive datasets derived from satellite imagery
and other remote sensing platforms.

## Cite Jeo

```  sh
@software{jeo2025:github,
  author = {Maxim Neumann and Anton Raichuk and Michelangelo Conserva and
  Luis Miguel Pazos-Outón and Keith Anderson and Matt Overlan and Mélanie Rey
  and Yuchang Jiang and Petra Poklukar and Cristina Nader Vasconcelos},
  title = {{JEO}: Model training and inference for geospatial remote sensing and
  {E}arth {O}bservation in {JAX}},
  url = {https://github.com/google-deepmind/jeo},
  year = {2025}
}
```
