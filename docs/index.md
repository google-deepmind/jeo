# Welcome to Jeo

<div style="text-align: left">
<img align="right" src="https://raw.githubusercontent.com/google-deepmind/jeo/main/docs/images/jeo_logo_1.png" width="100">
</div>

<div style="
  background-color: transparent;
  padding: 5px;
  text-align: center;
  color:rgb(66, 66, 66);
  font-weight: bold;
  font-size: 1.5em;
  font-style: italic;
  margin-left: auto;
  margin-right: auto;
  width: fit-content;
">
  Jax library to support machine learning research for remote sensing and Earth
  observation.
</div>

<br>

<div class="mdx-hero__content">
  <a href="https://github.com/google-deepmind/jeo" class="md-button">
    GitHub repository
  </a>
</div>

---

## Overview

Jeo is a specialized open source framework developed by Google DeepMind that
accelerates machine learning for geospatial remote sensing and earth observation
(EO) tasks. It uses [JAX](https://github.com/google/jax) and
[Flax](https://github.com/google/flax) for high-performance model training on
large geospatial datasets.

Jeo is tailored to the characteristics of geospatial datasets to help in the
development of models that can operate at scale. Jeo is primarily intended for
researchers and developers actively working in fields such as geospatial
analysis, remote sensing, environmental science, and sustainability modelling.

To access Jeo, visit [the GitHub repository](https://github.com/google-deepmind/jeo).

## Design

Jeo code structure is inspired by
[Big Vision](https://github.com/google-research/big_vision) and
[Scenic](https://github.com/google-research/scenic), and it builds upon the
following:

-   **[JAX](https://github.com/google/jax)**: provides the engine for
    high-performance computation.
-   **[Flax](https://github.com/google/flax)**: offers tools for building neural
    network models.
-   **[`tf.data`](https://www.tensorflow.org/guide/data)**: manages data input
    pipelines.
-   **[GeeFlow](https://github.com/google-deepmind/geeflow)**: connects the
    framework to [Google Earth Engine](https://earthengine.google.com/)'s data
    resources.

Familiarity with these components can aid in customizing and extending Jeo for
specific research needs. By using JAX and Flax, Jeo inherently benefits from
features like automatic differentiation, code compilation (XLA), and seamless
execution across various hardware accelerators, including CPUs, GPUs, and
[Google Cloud TPUs](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms).
This focus on performance is particularly relevant for earth observation tasks,
which often involve processing massive datasets derived from satellite imagery
and other remote sensing platforms.

Furthermore, Jeo's effective integration with libraries like GeeFlow enables
efficient construction of large-scale datasets directly from Google Earth Engine
(GEE), streamlining workflows in the GEE ecosystem. This combination makes Jeo a
potent tool for researchers pushing the boundaries of AI applications in
understanding and modelling our planet for research and sustainability projects.

## Cite Jeo

Cite the Jeo codebase as follows:

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
