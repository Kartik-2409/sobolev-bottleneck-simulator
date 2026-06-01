import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# --- 1. Mathematical Engine ---
def calculate_sobolev_energy(tensor):
    grad_x = torch.abs(tensor[:, :, :, :-1] - tensor[:, :, :, 1:])
    grad_y = torch.abs(tensor[:, :, :-1, :] - tensor[:, :, 1:, :])
    energy = torch.mean(grad_x) + torch.mean(grad_y)
    return energy.item()

def simulate_network_path(tensor, compression_factor, skip_engaged):
    original_size = tensor.shape[-2:]
    crushed = F.max_pool2d(tensor, kernel_size=compression_factor)
    reconstructed = F.interpolate(crushed, size=original_size, mode='nearest')

    if skip_engaged:
        high_frequency_edges = tensor - reconstructed
        final_output = reconstructed + high_frequency_edges
    else:
        final_output = reconstructed

    return crushed, final_output

# --- 2. The 3D Morphing Visualizer ---
def plot_3d_tensor(spatial_size, depth):
    s = spatial_size / 2
    x = [-s, s, s, -s, -s, s, s, -s]
    y = [-s, -s, s, s, -s, -s, s, s]
    z = [0, 0, 0, 0, depth, depth, depth, depth]
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

    fig = go.Figure(data=[
        go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='#00CC96', opacity=0.6, flatshading=True)
    ])
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-150, 150], title="Width"),
            yaxis=dict(range=[-150, 150], title="Height"),
            zaxis=dict(range=[0, 1200], title="Channels"),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1.5)
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=350
    )
    return fig

# --- 3. Streamlit UI Configuration ---
st.set_page_config(page_title="Sobolev Bottleneck Simulator", layout="wide")
st.title("The Sobolev Paradox in Deep Learning")
st.markdown("**An Interactive Proof of Spectral Bias and Memory Loss in Autoencoders.**")

st.markdown(r"<h2 style='text-align: center;'>$$| f |_{H^1}^2 = \int_{\Omega} |\nabla f(x)|^2 dx$$</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>The Sobolev semi-norm measuring high-frequency gradient energy.</p>", unsafe_allow_html=True)
st.divider()

# --- 4. Sidebar Controls ---
st.sidebar.header("Network Parameters")
compression = st.sidebar.slider("Bottleneck Compression Factor", min_value=1, max_value=32, value=1, step=1)
skip_engaged = st.sidebar.toggle("Engage Skip Connection", value=False)
uploaded_file = st.sidebar.file_uploader("Upload Test Image", type=["png", "jpg", "jpeg"])

# --- 5. Main Logic Pipeline ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    input_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    original_energy = calculate_sobolev_energy(input_tensor)
    if compression > 1:
        crushed_tensor, final_tensor = simulate_network_path(input_tensor, compression, skip_engaged)
    else:
        crushed_tensor, final_tensor = input_tensor, input_tensor

    final_energy = calculate_sobolev_energy(final_tensor)
    preservation_rate = (final_energy / original_energy) * 100 if original_energy > 0 else 100

    st.subheader("1. Spatial Compression vs. Mathematical Depth")
    col_3d, col_metrics = st.columns([2, 1])

    with col_3d:
        spatial_dim = 256 / compression
        depth_dim = 32 * compression
        st.plotly_chart(plot_3d_tensor(spatial_dim, depth_dim), use_container_width=True)

    with col_metrics:
        st.write("### Live Sobolev Analysis")
        st.metric("Input Gradient Energy", f"{original_energy:.4f}")
        st.metric("Output Gradient Energy", f"{final_energy:.4f}")

        if skip_engaged:
            st.metric("High-Frequency Preservation", f"{preservation_rate:.1f}%", "+ Rescued by Skip Connection")
        else:
            loss = 100 - preservation_rate
            st.metric("High-Frequency Preservation", f"{preservation_rate:.1f}%", f"- {loss:.1f}% Spectral Bias Loss")

    st.divider()
    st.subheader("2. The Visual Proof (Pixel Level)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original Input ($S_f$)**")
        st.image(img_array, use_container_width=True, clamp=True)

    with col2:
        st.markdown(f"**The Bottleneck ($K_f$)**")
        crushed_display = F.interpolate(crushed_tensor, size=(256, 256), mode='nearest').squeeze().numpy()
        st.image(crushed_display, use_container_width=True, clamp=True)

    with col3:
        if skip_engaged:
            st.markdown("**Reconstructed (With Skip Connection)**")
        else:
            st.markdown("**Reconstructed (Memory Loss)**")
        st.image(final_tensor.squeeze().numpy(), use_container_width=True, clamp=True)
else:
    st.info("👈 Upload an image in the sidebar to begin the mathematical diagnostic.")
