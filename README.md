# Jeo - Jax Geo lib

<div style="text-align: left">
<img align="right" src="https://raw.githubusercontent.com/google-deepmind/jeo/main/docs/images/jeo_logo_1.png" width="100" alt="jeo logo">
</div>

*Model training and inference for geospatial remote sensing and Earth
Observation in JAX.*

Jeo provides a framework for training machine learning models for geospatial
remote sensing and Earth Observation using [JAX](https://github.com/google/jax)
and [Flax](https://github.com/google/flax). It leverages
[tf.data](https://www.tensorflow.org/guide/data) for efficient data pipelines,
with a focus on [TensorFlow Datasets](https://www.tensorflow.org/datasets) for
scalable and reproducible input. While TensorFlow Datasets are preferred, other
dataset loaders are also supported. The codebase is designed to run seamlessly
on CPUs, GPUs, or
[Google Cloud TPU VMs](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms).

This project is open-sourced to share research code and facilitate collaboration
in geospatial and sustainability model development. Jeo integrates effectively
with the [GeeFlow](https://github.com/google-deepmind/geeflow) library to
construct large-scale geospatial datasets using
[Google Earth Engine](https://earthengine.google.com/) (GEE). An example
workflow is outlined below.

<div align="center">
<img src="https://raw.githubusercontent.com/google-deepmind/jeo/main/docs/images/jeo_geeflow_processing.png" width="900">
</div>

## Projects and publications

Projects and publications that used this codebase:

-   Light-weight geospatial model for global deforestation attribution, by
    *Anton Raichuk, Michelle Sims, Radost Stanimirova, and Maxim Neumann*.
    Presented at the
    [NeurIPS 2024 Workshop on Tackling Climate Change with Machine Learning](https://www.climatechange.ai/events/neurips2024)
    in Vancouver, BC, Canada. Dec 2024.
-   [Planted: a dataset for planted forest identification from multi-satellite
    time series](https://arxiv.org/abs/2406.18554), by *Luis Miguel Pazos-Outón,
    Cristina Nader Vasconcelos, Anton Raichuk, Anurag Arnab, Dan Morris, and
    Maxim Neumann*. Presented at [IGARSS 2024](https://www.2024.ieeeigarss.org/)
    in Athens, Greece. Jul 2024.
-   Global drivers of forest loss at 1 km resolution, by *Michelle Sims, Radost
    Stanimirova, Anton Raichuk, Maxim Neumann, Jessica Richter, Forrest Follett,
    James MacCarthy, Kristine Lister, Christopher Randle, Lindsey Sloat, Elena
    Esipova, Jaelah Jupiter, Charlotte Stanton, Dan Morris, Christy Slay, Drew
    Purves, and Nancy Harris*, 2025.
-   ForestCast: Forecasting deforestation risk at scale with deep learning, by
    *Matt Overlan, Charlotte Stanton, Maxim Neumann, Michelangelo Conserva,
    Yuchang Jiang, Arianna Manzini, Julia Haas, Mélanie Rey, Keith Anderson, and
    Drew Purves*, 2025.

## Getting started

### Installation

The first step is to checkout JEO and install relevant python dependencies in a
virtual environment:

```sh
git clone https://github.com/google-deepmind/jeo
# Go into the code directory for the examples below.
cd jeo/jeo
# Install and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
# Install JEO.
pip install -e ..
```

### Local demo run

Launching a quick local model training for just a few steps on the CPU:

```sh
python -m jeo.train --config configs/tests/tiny_bit.py:runlocal \
--workdir /tmp/jeo/demo_tiny_bit
```

This will start to train a `bit` model (which is a modified convolutional
neureal net (CNN) model based on ResNet, see `jeo/models/bit.py`) on
[CIFAR-10](https://www.tensorflow.org/datasets/catalog/cifar10) TFDS dataset.
Since we run this on a local CPU just as a demo, we appended the `:runlocal`
config arg above, which specifies within the config to run just for a few
training and evaluation steps. For more configuration details, see the config
file `jeo/configs/tests/tiny_bit.py`.

In a standard workflow, the given `workdir` will be used to save checkpoints and
potentially other artifacts, such as final evaluation metrics.

## Citing JEO

To cite this repository:

```
@software{jeo2025:github,
  author = {Maxim Neumann and Anton Raichuk and Michelangelo Conserva and
  Luis Miguel Pazos-Outón and Keith Anderson and Matt Overlan and Mélanie Rey
  and Yuchang Jiang and Petra Poklukar and Cristina Nader Vasconcelos},
  title = {{JEO}: Model training and inference for geospatial remote sensing and
  {E}arth {O}bservation in {JAX}}.
  url = {https://github.com/google-deepmind/jeo},
  year = {2025}
}
```

## License

Copyright 2024 DeepMind Technologies Limited

This code is licensed under the Apache License, Version 2.0 (the \"License\");
you may not use this file except in compliance with the License. You may obtain
a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an AS IS BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

## Disclaimer

This is not an official Google product.
