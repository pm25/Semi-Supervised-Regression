<a id="readme-top"></a>

# Benchmark Results

<details>
  <summary><strong>📋 Table of Contents</strong></summary>
  <ol>
    <li><a href="#-classic-cv-results">Classic CV Results</a></li>
    <li><a href="#-nlp-results">NLP Results</a></li>
    <li><a href="#-audio-results">Audio Results</a></li>
  </ol>
</details>

This folder contains benchmark results and experiment logs for semi-supervised regression algorithms. The best evaluation metrics (e.g., MAE, MSE, R², SRCC) are reported across training iterations. Each setting runs 3 different random seeds and computes the average performance with standard deviation.

## 📷 Classic CV Results

### a.) Datasets

| Dataset | # Labels        | # Training Data | # Test Data | Target Range |
| ------- | --------------- | --------------- | ----------- | ------------ |
| UTKFace | 50 / 250 / 2000 | 18,964          | 4,741       | [1, 116]     |

### b.) Results

The results for Classic CV benchmarks can be found in [classic_cv_average_log.csv](classic_cv_average_log.csv).

### c.) Logs

For detailed experiment logs, check out our **WandB project page** (🚧 Coming Soon).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📝 NLP Results

### a.) Datasets

| Dataset     | # Labels | # Training Data | # Test Data | Target Range |
| ----------- | -------- | --------------- | ----------- | ------------ |
| Yelp Review | 250      | 250,000         | 25,000      | [0, 4]       |

### b.) Results

The results for NLP benchmarks can be found in [nlp_average_log.csv](nlp_average_log.csv).

### c.) Logs

For detailed experiment logs, check out our **WandB project page** (🚧 Coming Soon).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🎧 Audio Results

### a.) Datasets

| Dataset | # Labels | # Training Data | # Test Data | Target Range |
| ------- | -------- | --------------- | ----------- | ------------ |
| BVCC    | 250      | 4,974           | 1,066       | [1, 5]       |

### b.) Results

The results for Audio benchmarks can be found in [audio_average_log.csv](audio_average_log.csv).

### c.) Logs

For detailed experiment logs, check out our **WandB project page** (🚧 Coming Soon).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
