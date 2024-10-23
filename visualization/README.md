<a id="readme-top"></a>

<h1 id="t-sne-visualization" align="center">Visualization Results</h1>

<!-- TABLE OF CONTENTS -->
<details>
  <summary><strong>📋 Table of Contents</strong></summary>
  <ol>
    <li>
      <a href="#t-sne-visualization">t-SNE Visualization</a>
    </li>
    <li>
      <a href="#umap-visualization">UMAP Visualization</a>
    </li>
  </ol>
</details>

Below are examples of feature visualizations using t-SNE and UMAP on various semi-supervised regression methods. These visualizations are generated from the test data of the UTKFace dataset, with a setting of 250 labeled samples. Both 2D and 3D versions are provided.

<hr>

<h2 id="t-sne-visualization" align="center">✨ t-SNE Visualization ✨</h2>

<p align="center">
    <strong>Parameters</strong>: 
    <br> 
    <strong>Perplexity</strong>: 50 &nbsp; | &nbsp; 
    <strong>Learning Rate</strong>: "auto" &nbsp; | &nbsp; 
    <strong>Initialization</strong>: PCA
</p>

|                                                                                                                                                                                                                                                   |                                                                                                                                                                                                                                                                     |                                                                                                                                                                                                                                                   |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   <img width="1604" alt="Supervised t-SNE" src="https://pm25.github.io/Research-Figures/rankup/visualization/supervised_t-sne_2d.png"> Supervised [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/supervised_t-sne_3d.html)]   |                  <img width="1604" alt="Π-Model t-SNE" src="https://pm25.github.io/Research-Figures/rankup/visualization/pimodel_t-sne_2d.png"> Π-Model [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/pimodel_t-sne_3d.html)]                  | <img width="1604" alt="MeanTeacher t-SNE" src="https://pm25.github.io/Research-Figures/rankup/visualization/meanteacher_t-sne_2d.png"> MeanTeacher [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/meanteacher_t-sne_3d.html)] |
|               <img width="1604" alt="CLSS t-SNE" src="https://pm25.github.io/Research-Figures/rankup/visualization/clss_t-sne_2d.png"> CLSS [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/clss_t-sne_3d.html)]               |                <img width="1604" alt="MixMatch t-SNE" src="https://pm25.github.io/Research-Figures/rankup/visualization/mixmatch_t-sne_2d.png"> MixMatch [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/mixmatch_t-sne_3d.html)]                |           <img width="1604" alt="RankUp t-SNE" src="https://pm25.github.io/Research-Figures/rankup/visualization/rankup_t-sne_2d.png"> RankUp [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/rankup_t-sne_3d.html)]           |
| <img width="1604" alt="RankUp + RDA t-SNE" src="https://pm25.github.io/Research-Figures/rankup/visualization/rankup_rda_t-sne_2d.png"> RankUp + RDA [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/rankup_rda_t-sne_3d.html)] | <img width="1604" alt="Fully-Supervised t-SNE" src="https://pm25.github.io/Research-Figures/rankup/visualization/fullysupervised_t-sne_2d.png"> Fully-Supervised [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/fullysupervised_t-sne_3d.html)] |                                                                                                                                                                                                                                                   |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<h2 id="umap-visualization" align="center">✨ UMAP Visualization ✨</h2>

<p align="center">
    <strong>Parameters</strong>: 
    <br> 
    <strong>Neighbors</strong>: 50 &nbsp; | &nbsp; 
    <strong>Initialization</strong>: PCA
</p>

|                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                                  |                                                                                                                                                                                                                                                |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   <img width="1604" alt="Supervised UMAP" src="https://pm25.github.io/Research-Figures/rankup/visualization/supervised_umap_2d.png"> Supervised [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/supervised_umap_3d.html)]   |                  <img width="1604" alt="Π-Model UMAP" src="https://pm25.github.io/Research-Figures/rankup/visualization/pimodel_umap_2d.png"> Π-Model [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/pimodel_umap_3d.html)]                  | <img width="1604" alt="MeanTeacher UMAP" src="https://pm25.github.io/Research-Figures/rankup/visualization/meanteacher_umap_2d.png"> MeanTeacher [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/meanteacher_umap_3d.html)] |
|               <img width="1604" alt="CLSS UMAP" src="https://pm25.github.io/Research-Figures/rankup/visualization/clss_umap_2d.png"> CLSS [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/clss_umap_3d.html)]               |                <img width="1604" alt="MixMatch UMAP" src="https://pm25.github.io/Research-Figures/rankup/visualization/mixmatch_umap_2d.png"> MixMatch [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/mixmatch_umap_3d.html)]                |           <img width="1604" alt="RankUp UMAP" src="https://pm25.github.io/Research-Figures/rankup/visualization/rankup_umap_2d.png"> RankUp [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/rankup_umap_3d.html)]           |
| <img width="1604" alt="RankUp + RDA UMAP" src="https://pm25.github.io/Research-Figures/rankup/visualization/rankup_rda_umap_2d.png"> RankUp + RDA [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/rankup_rda_umap_3d.html)] | <img width="1604" alt="Fully-Supervised UMAP" src="https://pm25.github.io/Research-Figures/rankup/visualization/fullysupervised_umap_2d.png"> Fully-Supervised [[3D](https://pm25.github.io/Research-Figures/rankup/visualization/fullysupervised_umap_3d.html)] |                                                                                                                                                                                                                                                |

<p align="right">(<a href="#readme-top">back to top</a>)</p>
