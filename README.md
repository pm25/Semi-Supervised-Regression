<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<a id="readme-top"></a>

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->

<!-- <br /> -->
<!-- PROJECT LOGO -->
  <!-- <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

<h1 align="center">🎋 Semi-Supervised Regression</h1>

<p align="center">
  Welcome to the official repository for the paper <br />
  <strong>"RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier"</strong>,<br />
  by Pin-Yen Huang, Szu-Wei Fu, and Yu Tsao. Accepted at NeurIPS 2024.
</p>

<p align="center">
  Explore the following resources:
  <br />
  <a href="./results/README.md">🗃️ Experiment Logs</a>
  ·
  <a href="./visualization/README.md">✨ Feature Visualizations</a>
</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary><strong>📋 Table of Contents</strong></summary>
  <ol>
    <li><a href="#1-introduction">Introduction</a></li>
    <li><a href="#2-getting-started">Getting Started</a></li>
    <li><a href="#3-usage">Usage</a></li>
    <li><a href="#4-benchmark-results">Benchmark Results</a></li>
    <li><a href="#5-license">License</a></li>
    <li><a href="#6-contact">Contact</a></li>
    <li><a href="#7-citing-this-work">Citing This Work</a></li>
    <li><a href="#8-acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## 1. Introduction

[![RankUp Illustration][rankup-illustration]](https://github.com/pm25/semi-supervised-regression)

<!-- -   Paper Link: []() -->

-   Github Link: [https://github.com/pm25/semi-supervised-regression](https://github.com/pm25/semi-supervised-regression)

This repository contains the official code and documentation for the paper _"RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier"_. Here, you’ll find implementations of the proposed methods (RankUp, ARC, and RDA), alongside related works, experiment logs, feature visualizations, datasets, and hyperparameter settings used in our experiments. The code is adapted from the popular semi-supervised classification framework, [USB](https://github.com/microsoft/semi-supervised-learning). We are deeply grateful to the USB team for their excellent framework, which played a crucial role in the development of our research and made our work possible!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## 2. Getting Started

This section guides you through setting up the repository locally. Follow these steps to get your environment ready.

### 💽 Clone Repository

Clone this repository with the following command:

```bash
git clone https://github.com/pm25/semi-supervised-regression.git
```

### 🪛 Prerequisites

This repository is built on PyTorch, along with `torchvision`, `torchaudio`, and `transformers`.

To install the required packages, first create a Conda environment:

```bash
conda create --name ssr python=3.8
```

Activate the environment:

```bash
conda activate ssr
```

Then, use `pip` to install the required packages:

```bash
python -m pip install -r requirements.txt
```

Once the installation is complete, you can start using this repository by running:

```bash
python train.py --gpu 0 --c config/classic_cv/rankup/supervised_fixmatch_utkface_lb250_s0.yaml
```

### ⚙️ Configuration Files

All configuration files are stored in the [config](./config) folder.

The configuration file paths follow this naming format:

```python
./config/{data_type}/{method}/{regression_method}_{classification_method}_{dataset}_lb{num_labeled}_s{seed}.yaml
```

For example, the configuration file for the RankUp method (combining RDA and FixMatch) on the UTKFace dataset with 250 labeled samples and seed 0 is located at:

```bash
./config/classic_cv/rankup/rda_fixmatch_utkface_lb250_s0.yaml
```

### 📊 Prepare Datasets

No manual dataset preparation is required. The datasets will be automatically downloaded and processed when you run the code. For more details on the dataset splits and processing, visit the [pm25/Regression-Datasets](https://github.com/pm25/regression-datasets).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## 3. Usage

This repository is designed to be user-friendly and extendable. Below are some examples to help you get started with training, evaluation, and feature visualization.

### 🏋️ Training

To train RankUp on the UTKFace dataset with 250 labels on GPU 0, use the following command. You can modify the configuration file to try other algorithms, datasets, or label settings:

```bash
python train.py --gpu 0 --c config/classic_cv/rankup/supervised_fixmatch_utkface_lb250_s0.yaml
```

### 🎯 Evaluation

After training, you can either view the evaluation results in the training logs or run the evaluation script:

```bash
python eval.py --gpu 0 --c config/classic_cv/rankup/supervised_fixmatch_utkface_lb250_s0.yaml
```

### ✨ Features Visualization

Once training is complete, you can extract hidden features of the evaluation data using:

```bash
python eval.py --gpu 0 --save_features \
    --c config/classic_cv/rankup/supervised_fixmatch_utkface_lb250_s0.yaml
```

To visualize these features, you can project them into 2D or 3D space using t-SNE or UMAP:

```bash
python visualization/plot.py --methods tsne umap --output_dim 2 \
    --load_path visualization/features/supervised_fixmatch_utkface_lb250_s0.npy
```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- BENCHMARK RESULTS -->

## 4. Benchmark Results

Please refer to [Results](./results) for benchmark results and experiment logs on different tasks and labeled settings.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## 5. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## 6. Contact

-   Pin-Yen Huang (pyhuang97@gmail.com), Arizona State University
-   Szu-Wei Fu (szuweif@nvidia.com), NVIDIA
-   Yu Tsao (yu.tsao@citi.sinica.edu.tw), Academia Sinica

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Citing -->

## 7. Citing this Work

If you find this repository useful, please consider citing our paper or giving this repository a star. Your support is greatly appreciated!

```
@inproceedings{,
  doi = {},
  url = {},
  author = {Pin-Yen Huang, Szu-Wei Fu and Yu Tsao},
  title = {RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier},
  booktitle = {}
  year = {2024}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## 8. Acknowledgments

-   [USB](https://github.com/microsoft/semi-supervised-learning)
-   [TorchSSL](https://github.com/TorchSSL/TorchSSL)
-   [README Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[linkedin-url]: https://linkedin.com/in/py-huang
[rankup-illustration]: figures/rankup-illustration.png
