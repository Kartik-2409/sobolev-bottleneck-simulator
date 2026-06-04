# Spectral Bias & The Sobolev Paradox in Deep Learning
**An Interactive Diagnostic Tool for Autoencoder Topologies**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sobolev-bottleneck-simulator-8uhx73cwcxo9csshvmrjqr.streamlit.app/)

This repository contains a Streamlit-based interactive simulator developed as part of a Directed Independent Study in Mathematics (Neural Network Operators in Approximation Theory) at SVNIT. 

The tool visually and mathematically demonstrates the collapse of the $H^1$ Sobolev semi-norm (gradient energy) during spatial downsampling in autoencoders, and proves the topological necessity of residual skip connections to preserve high-frequency boundary conditions.

### Author
**Kartik R C Pilley** *Integrated M.Sc. Mathematics*

---

## 📊 Dashboard Overview
![Sobolev Simulator UI](assets/ui_overview.png)

## 🧠 The Intuition (In Plain English)
If you are new to approximation theory, here is exactly what this topology is doing, conceptualized as painting a portrait:

**1. The Bottleneck (The Smudged Canvas)**
When an image passes through a deep neural network bottleneck, the spatial dimensions are violently compressed via Max Pooling. This is like blocking out the basic colors and proportions of a face on a canvas, but using a heavy smudge tool. You gain deep structural understanding (where the eyes and nose belong), but you permanently erase the high-frequency micro-textures (the individual eyelashes, the sharp edge of the pupil). The image becomes a blurry map.

**2. The Skip Connection (The Tracing Paper Overlay)**
If the network outputs that smudged map, the resolution is ruined. A skip connection solves this by acting as a mathematical bypass wire. 

Imagine taking a piece of tracing paper and copying the exact, hyper-realistic outlines of the eyes and skin texture *before* you started smudging the canvas. The skip connection routes this "tracing paper" completely around the destructive bottleneck and glues it directly on top of the final blurry output. By combining the deep conceptual mapping of the bottleneck with the bypassed high-frequency spatial edges, the network achieves perfect structural reconstruction.

---

## 🔬 The Visual Proof (Topological Reconstruction)
![Reconstruction Output](assets/reconstruction_proof.jpg)
*At a compression kernel size of 7, the bottleneck ($K_f$) completely loses the high-frequency boundaries of the input data. Engaging the skip connection mathematically bypasses the bottleneck, resulting in 100% gradient energy preservation and a flawless topological reconstruction.*

---

## 📐 Mathematical Formulation & Proofs
For a rigorous mathematical breakdown of the Sobolev bottleneck, finite difference gradient calculations, and the spectral bias proof, please read the accompanying documentation:
👉 **[Read the Mathematical Theory (THEORY.md)](Theory.md)**

---

## 🚀 Running the Simulator Locally

### Prerequisites
Ensure you have Python 3.8+ installed. The environment requires PyTorch, Streamlit, and Plotly.

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/sobolev-paradox-sim.git](https://github.com/yourusername/sobolev-paradox-sim.git)
   cd sobolev-paradox-sim
