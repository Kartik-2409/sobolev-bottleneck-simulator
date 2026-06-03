"""
Description:
A diagnostic tool demonstrating the collapse of the H1 semi-norm (Sobolev gradient energy) 
during spatial downsampling in autoencoder architectures, and the necessity of high-frequency 
skip connections to preserve boundary conditions.
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import plotly.graph_objects as go

def calc_h1_seminorm(tensor):
    """
    Approximates the H1 semi-norm of a 2D tensor using discrete finite differences.
    This acts as our metric for high-frequency texture preservation.
    """
    # Calculate gradients along X and Y axes
    grad_x = torch.abs(tensor[:, :, :, :-1] - tensor[:, :, :, 1:])
    grad_y = torch.abs(tensor[:, :, :-1, :] - tensor[:, :, 1:, :])
    
    # L2 integral approximation over the domain
    energy = torch.mean(grad_x) + torch.mean(grad_y)
    return energy.item()

def forward_pass_sim(tensor, pool_kernel, use_skip):
    """Simulates the compression bottleneck and optional residual bypass."""
    orig_dim = tensor.shape[-2:]
    
    # Low-pass filter bottleneck (Max Pooling)
    k_f = F.max_pool2d(tensor, kernel_size=pool_kernel)
    
    # Decoder upsampling (Nearest neighbor interpolation)
    reconstructed = F.interpolate(k_f, size=orig_dim, mode='nearest')
    
    if use_skip:
        # Extract high-frequency bounds and bypass the bottleneck
        hf_residual = tensor - reconstructed
        output = reconstructed + hf_residual
    else:
        output = reconstructed
        
    return k_f, output

def render_tensor_topology(spatial_dim, channel_depth):
    """Renders the physical tensor deformation via Plotly."""
    s = spatial_dim / 2
    x = [-s, s, s, -s, -s, s, s, -s]
    y = [-s, -s, s, s, -s, -s, s, s]
    z = [0, 0, 0, 0, channel_depth, channel_depth, channel_depth, channel_depth]
    
    # Mesh geometry
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='#00CC96', opacity=0.6, flatshading=True)])
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-150, 150], title="Spatial (X)"),
            yaxis=dict(range=[-150, 150], title="Spatial (Y)"),
            zaxis=dict(range=[0, 1200], title="Features (Z)"),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1.5)
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=350
    )
    return fig

# UI Setup
st.set_page_config(page_title="Sobolev Bottleneck Analysis", layout="wide")
st.title("Spectral Bias & The Sobolev Paradox in Deep Learning")

# Mathematical Formulation (Using pure markdown for stable MathJax rendering)
st.markdown(r"$$| f |_{H^1}^2 = \int_{\Omega} |\nabla f(x)|^2 dx$$")
st.markdown("<p style='text-align: center; color: gray;'>The Sobolev semi-norm measuring high-frequency boundary energy.</p>", unsafe_allow_html=True)
st.divider()

# Sidebar config
st.sidebar.header("Architecture Parameters")
kernel_size = st.sidebar.slider("Pooling Kernel Size (Downsampling)", min_value=1, max_value=32, value=1, step=1)
skip_connection = st.sidebar.toggle("Inject High-Frequency Bypass (Skip)", value=False)
upload = st.sidebar.file_uploader("Input Data (Image)", type=["png", "jpg", "jpeg"])

if upload is not None:
    # Preprocess image to tensor
    raw_img = Image.open(upload).convert('L').resize((256, 256))
    img_matrix = np.array(raw_img) / 255.0
    u_tensor = torch.tensor(img_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Execute operators
    initial_energy = calc_h1_seminorm(u_tensor)
    if kernel_size > 1:
        bottleneck_tensor, final_tensor = forward_pass_sim(u_tensor, kernel_size, skip_connection)
    else:
        bottleneck_tensor, final_tensor = u_tensor, u_tensor
        
    final_energy = calc_h1_seminorm(final_tensor)
    preservation = (final_energy / initial_energy) * 100 if initial_energy > 0 else 100

    # Dashboard Metrics
    st.subheader("1. Spatial Compression vs. Feature Depth")
    col_plot, col_stats = st.columns([2, 1])
    
    with col_plot:
        s_dim = 256 / kernel_size
        c_dim = 32 * kernel_size
        st.plotly_chart(render_tensor_topology(s_dim, c_dim), use_container_width=True)
        
    with col_stats:
        st.write("### H1 Norm Analysis")
        st.metric("Input Gradient Energy", f"{initial_energy:.4f}")
        st.metric("Output Gradient Energy", f"{final_energy:.4f}")
        
        if skip_connection:
            st.metric("Boundary Preservation", f"{preservation:.1f}%", "+ Rescued via Skip Connection")
        else:
            st.metric("Boundary Preservation", f"{preservation:.1f}%", f"- {100 - preservation:.1f}% Energy Loss")

    st.divider()
    st.subheader("2. Topological Reconstruction")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Original Function ($f$)**")
        st.image(img_matrix, use_container_width=True, clamp=True)
        
    with c2:
        st.markdown(f"**Bottleneck ($K_f$)**")
        k_display = F.interpolate(bottleneck_tensor, size=(256, 256), mode='nearest').squeeze().numpy()
        st.image(k_display, use_container_width=True, clamp=True)
        
    with c3:
        st.markdown("**Reconstruction**")
        st.image(final_tensor.squeeze().numpy(), use_container_width=True, clamp=True)
else:
    st.info("Awaiting input data...")
