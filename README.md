# Pokémon Species Identification via Fine-Grained Visual Categorization

**Course: DS 6050: Deep Learning**

Fine-grained visual categorization (FGVC) presents a distinct challenge in computer vision: distinguishing between sub-categories that exhibit high inter-class similarity and significant intra-class variance. This project evaluates these challenges through Pokémon species identification, utilizing a dataset characterized by minute ornamental features and diverse artistic media, ranging from 2D pixel art to 3D renderings. Unlike standard benchmarks such as CIFAR-10, this high-cardinality task requires models to maintain robustness across varying backgrounds and abstraction levels. By successfully classifying 721 unique species, we aim to demonstrate the portability of these architectures to biological conservation tasks, such as identifying endangered species in inconsistent field environments.

# Research Question: 
To what extent does a hybrid Inception-ResNet-v2 architecture, utilizing multi-scale feature extraction and residual learning, outperform standard residual networks and attention-based Vision Transformers in the fine-grained classification of stylized, multi-modal artistic datasets?

# Current literature: 
- Highlights a progression from limited-scope CNNs to advanced hybrid architectures
1. Scope vs. Scalability: Early work by Rawat (2021) achieved 96% accuracy but was restricted to 151 species. Our study expands this to 721 species to test scalability.
2. Multimodal Challenges: Zahroof (2019) emphasized that model robustness often fails when transitioning between stylized sprites and 3D renderings, a gap this project seeks to bridge.
3. Data Scarcity: Li et al. (2019) introduced Relation Networks to address "long-tail" distributions where rare species have limited training data.
4. Architectural Benchmarks: Building on Saxena et al. (2025), we utilize Inception-ResNet-v2 to validate its effectiveness in capturing multi-scale visual features.

## Technical Approach:
The core architecture is an Inception-ResNet-v2 leveraging transfer learning. Inception modules extract multi-scale features (local textures and global shapes) in parallel, while residual connections ensure stable gradient flow during deep training.

**Training Configuration**
1. Loss Function: Categorical Cross-Entropy.
2. Optimizer: Adam (Adaptive Moment Estimation) for faster convergence.
3. Regularization: Batch normalization, dropout, and early stopping to mitigate overfitting.

**Ablation Studies**
To ensure rigorous research as required by the DS 6050 curriculum, we conduct the following experiments:
1. Backbone Comparison: Replacing the hybrid backbone with a standard ResNet-50 and a Vision Transformer to isolate architectural benefits.
2. Pre-training Impact: Comparing the pre-trained model against a model trained from scratch to quantify the value of transfer learning.
3. Augmentation Sensitivity: Disabling transformations (rotation, flipping) to determine the model's reliance on orientation-invariant features.

# Dataset
- Source: [Pokemon Image Dataset](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types) (Publicly available)
- Composition: 10,073 images across 721 species.
- Protocol: 80/20 train-test split with episodic validation. All training is conducted on UVA Rivanna HPC resources.

# References
- Saxena, R.R., et al. (2025). Pokémondium: A Machine Learning Approach to Detecting Images of Pokémon.
- Li, X., et al. (2019). One-shot Pokemon Classification using Relation Networks.
- Rawat, A. (2021). Pokémon Classification Using CNN.
- Zahroof, T. (2019). What's That Pokemon? Stanford University CS230.
