# Probabilistic-SAM
This repository contains the implementation of Probabilistic SAM. Our model learns a latent variable space that captures uncertainty and annotator variability in medical images. At inference, the model samples from this latent space, producing diverse masks that reflect the inherent ambiguity in medical image segmentation.

## Abstract
Recent advances in promptable segmentation, such as the Segment Anything Model (SAM), have enabled flexible, high-quality mask generation across a wide range of visual domains. However, SAM and similar models remain fundamentally deterministic, producing a single segmentation per object per prompt, and fail to capture the inherent ambiguity present in many real-world tasks. This limitation is particularly troublesome in medical imaging, where multiple plausible segmentations may exist due to annotation uncertainty or inter-expert variability. In this paper, we introduce Probabilistic SAM, a probabilistic extension of SAM that models a distribution over segmentations conditioned on both the input image and prompt. By incorporating a latent variable space and training with a variational objective, our model learns to generate diverse and plausible segmentation masks reflecting the variability in human annotations. The architecture integrates a prior and posterior network into the SAM framework, allowing latent codes to modulate the prompt embeddings during inference. The latent space allows for efficient sampling during inference, enabling uncertainty-aware outputs with minimal overhead. We evaluate Probabilistic SAM on the LIDC-IDRI lung nodule dataset and demonstrate its ability to produce diverse outputs that align with expert disagreement, outperforming existing probabilistic baselines on uncertainty-aware metrics.

## Model
![Figure](https://github.com/tbwa233/Probabilistic-SAM/blob/main/images/probsam_training.png)

Given a CT slice and a bounding box prompt $(x_1, y_1), (x_2, y_2)$, visual and spatial information is encoded via SAM's image and prompt encoders. During training, a posterior network uses image embeddings and the ground truth mask to estimate $\mathcal{N}(\mu_{\text{post}}, \sigma_{\text{post}})$, while a prior network predicts $\mathcal{N}(\mu_{\text{prior}}, \sigma_{\text{prior}})$. A latent vector $z \sim \mathcal{N}(\mu_{\text{post}}, \sigma_{\text{post}})$ sampled from the posterior network is projected and added to the prompt embeddings before decoding. The model is optimized using a combination of binary cross-entropy (BCE), Dice loss, and Kullback–Leibler (KL) divergence between the posterior and prior distributions.

![Figure](https://github.com/tbwa233/Probabilistic-SAM/blob/main/images/probsam_sampling.png)

A prior network maps image embeddings to a Gaussian latent space, from which latent vectors $z_1, z_2, z_3, \dots$ are sampled. After projection through a multilayer perceptron (MLP), these vectors are added to the sparse prompt embeddings. The modified prompts and image embeddings are passed to SAM's lightweight mask decoder to generate diverse segmentation predictions.

## Results
A brief summary of our results are shown below. Our Probabilistic SAM is compared to various baselines. In the table, the best scores are bolded and the second-best scores are italicized.

| Model              | GED (↓)   | DSC (↑)   | IoU (↑)   |
|--------------------|-----------|-----------|-----------|
| Dropout U-Net      | 0.5156    | 0.5591    | 0.3880    |
| Dropout SAM        | 0.5025    | _0.6799_  | 0.5150    |
| Probabilistic U-Net| _0.3349_  | 0.5818    | _0.5557_  |
| Probabilistic SAM  | **0.2910**| **0.8255**| **0.7849**|

## Data
We evaluate Probabilistic SAM on the task of lung nodule segmentation using the [LIDC-IDRI](https://pmc.ncbi.nlm.nih.gov/articles/PMC3041807/) dataset. This dataset contains thoracic CT scans along with ground truth annotations from four expert radiologists.

## Code
The code has been written in Python using the Pytorch framework. Training requries a GPU. To train your own Probabilistic SAM, simply clone this repository and run train.py.

## Acknowledgements
Thanks to [Stefan Knegt](https://github.com/stefanknegt) for open-sourcing his [Pytorch implementation of Probabilistic U-Net](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch), which served as a helpful guide in the development of Probabilistic SAM, and for providing a link to [pre-processed LIDC-IDRI data](https://drive.google.com/drive/folders/1xKfKCQo8qa6SAr3u7qWNtQjIphIrvmd5).
