# Variational Autoencoder for Smile Manipulation

## Overview

This project implements a Variational Autoencoder (VAE) using PyTorch and the CelebA dataset to learn latent representations of facial images and perform attribute-based image manipulation.

The model learns a meaningful latent space and modifies the smiling attribute of human faces by identifying and manipulating a smile direction vector.

---

## Project Objectives

- Learn compact latent representations of facial images
- Extract a semantic smile vector from latent space
- Generate images with increased or decreased smile intensity
- Demonstrate controllable image generation using deep learning

---

## Dataset

**CelebA (CelebFaces Attributes Dataset)**

- 200,000+ celebrity face images
- 40 annotated facial attributes
- Images resized to 64 × 64 pixels
- Smiling attribute used for latent space manipulation

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Computer Vision
- Deep Learning
- Variational Autoencoders (VAE)

---

## Methodology

### Encoder

Maps input facial images into a latent representation.

### Decoder

Reconstructs images from latent vectors.

### Loss Function

- Reconstruction Loss (MSE)
- KL Divergence

### Smile Vector Computation

The latent representations of smiling and non-smiling images are averaged.

A smile vector is computed as:

Smile Vector = Mean(Smiling Faces) − Mean(Non-Smiling Faces)

This vector is then added or subtracted from a latent representation to control smile intensity.

---

## Results

### Original Reconstruction

<img src="images/original_reconstructed_v2.png" width="300">

### Increased Smile

<img src="images/more_smiley_v2.png" width="300">

### Decreased Smile

<img src="images/less_smiley_v2.png" width="300">

---

## Key Findings

- Successfully learned meaningful latent facial representations
- Controlled facial attributes through latent space manipulation
- Generated smooth changes in smile intensity
- Demonstrated interpretable representation learning

---

## Future Improvements

- Beta-VAE implementation
- GAN-based image enhancement
- Higher resolution image generation
- Multi-attribute manipulation (age, glasses, gender)

---

## Repository Contents

```text
project2.py          # Model implementation
Project_Report.pdf   # Detailed project report
User_Manual.pdf      # User guide
images/              # Output visualizations
```

---

## Author

**Zakariyya Shahid**

MS in Business Analytics  
Simon Business School, University of Rochester
