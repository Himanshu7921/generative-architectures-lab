# Diffusion Generative Models — From Scratch

This folder contains a collection of diffusion-based generative models implemented **from scratch in PyTorch**, with the primary focus on understanding the mathematical foundations behind each method rather than treating the models as black-box architectures.

Each implementation is accompanied by an **end-to-end mathematical derivation**, covering the probabilistic formulation, training objective, sampling procedure, and the architectural components required to translate the mathematics into working code.

The goal is to progressively trace the evolution of modern diffusion and flow-based generative modeling, starting from the original DDPM formulation and moving toward continuous-time and flow-matching approaches.

---

## Models

### 1. DDPM — Denoising Diffusion Probabilistic Models

The foundational discrete diffusion formulation.

Topics covered:

- Forward diffusion process
- Markov formulation
- Gaussian transition kernels
- Closed-form forward process
- Reverse diffusion process
- Posterior derivation
- Variational lower bound (ELBO)
- ELBO decomposition
- Noise-prediction objective
- Forward and reverse sampling
- U-Net denoising architecture

---

### 2. DDIM — Denoising Diffusion Implicit Models

An alternative sampling formulation built on the DDPM training framework.

Topics covered:

- DDPM → DDIM formulation
- Non-Markovian forward process
- Deterministic sampling
- Stochastic sampling
- Reduced sampling steps
- DDIM sampling equation
- Relationship between DDPM and DDIM
- Effect of the stochasticity parameter

---

### 3. Classifier-Free Guidance (CFG)

Conditional generation without requiring a separate classifier during sampling.

Topics covered:

- Conditional diffusion
- Unconditional diffusion
- Conditioning dropout
- Conditional and unconditional score/noise predictions
- Guidance formulation
- Guidance scale
- Training and sampling procedure
- Effect of guidance on generation

---

### 4. Latent Diffusion Models (LDM)

Diffusion performed in a learned latent representation rather than directly in pixel space.

Topics covered:

- Image autoencoders
- Latent representations
- Diffusion in latent space
- Latent-space noise prediction
- Decoder reconstruction
- Computational motivation for latent diffusion
- Conditioning mechanisms
- Relationship between pixel-space and latent-space diffusion

---

### 5. DiT — Diffusion Transformers

Replacing the conventional convolutional U-Net denoising backbone with a Transformer architecture.

Topics covered:

- Image patchification
- Patch embeddings
- Transformer blocks
- Self-attention
- Timestep conditioning
- Adaptive LayerNorm conditioning
- Diffusion Transformer architecture
- Training and sampling

---

### 6. Score-Based Generative Models

A continuous-time perspective on diffusion through score estimation and stochastic differential equations.

Topics covered:

- Score function
- Denoising score matching
- Noise perturbation
- Variance-preserving SDEs
- Variance-exploding SDEs
- Forward SDE
- Reverse-time SDE
- Score estimation
- Probability-flow ODE
- Numerical sampling

---

### 7. Flow Matching & Rectified Flow

An alternative generative modeling framework based on learning a continuous velocity field between probability distributions.

Topics covered:

- Probability paths
- Conditional probability paths
- Velocity fields
- Flow matching objective
- Optimal transport intuition
- ODE-based generation
- Rectified flow
- Straightening probability trajectories
- Relationship between diffusion and flow-based formulations

---

## Repository Structure

```text
09_Diffusions_models/
│
├── README.md
│
├── 01_DDPM/
│   ├── derivation/
│   ├── notebook/
│   ├── src/
│   ├── README.md
│   ├── configs/
│   └── experiments/
│
├── 02_DDIM/
│   ├── derivation/
│   ├── notebook/
│   ├── src/
│   ├── README.md
│   ├── configs/
│   └── experiments/
│
├── 03_CFG/
│   ├── derivation/
│   ├── notebook/
│   ├── src/
│   ├── README.md
│   ├── configs/
│   └── experiments/
│
├── 04_LDM/
│   ├── derivation/
│   ├── notebook/
│   ├── src/
    ├── README.md
│   ├── configs/
│   └── experiments/
│
├── 05_DiT/
│   ├── derivation/
│   ├── notebook/
│   ├── src/
    ├── README.md
│   ├── configs/
│   └── experiments/
│
├── 06_Score_SDE/
│   ├── derivation/
│   ├── notebook/
│   ├── src/
    ├── README.md
│   ├── configs/
│   └── experiments/
│
└── 07_Flow_Matching/
    ├── derivation/
│   ├── notebook/
    ├── src/
    ├── configs/
    └── experiments/