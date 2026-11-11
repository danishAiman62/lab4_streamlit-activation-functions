import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Activation Visualiser", layout="centered")

st.title("⚡ ReLU Activation Function Visualiser")
st.markdown("**Student ID:** SD24008 | **Subject:** BSD3513 Introduction to AI")

st.write("""
The **Rectified Linear Unit (ReLU)** is the most widely used activation function in modern neural networks. 
It outputs the input directly if it is positive; otherwise, it outputs zero.
""")

# Controls
x_range = st.slider("Select Input Range (x)", -10.0, 10.0, (-5.0, 5.0), 0.5)
show_derivative = st.checkbox("Show Derivative / Gradient", value=True)

# Generate data
x = np.linspace(x_range[0], x_range[1], 400)
y = np.maximum(0, x)
dy = np.where(x > 0, 1.0, 0.0)

# Plotting
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(x, y, label="ReLU: f(x) = max(0, x)", color="#d95f02", linewidth=3)
if show_derivative:
    ax.plot(x, dy, label="Derivative: f'(x)", color="#7570b3", linestyle="--", linewidth=2)

ax.axhline(0, color='black', linewidth=1, linestyle='-')
ax.axvline(0, color='black', linewidth=1, linestyle='-')
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_title("Rectified Linear Unit (ReLU) & Gradient", fontsize=12, fontweight='bold')
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.legend()
st.pyplot(fig)

st.subheader("Mathematical Properties")
st.latex(r"f(x) = \max(0, x)")
st.latex(r"f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}")
st.info("**Key Benefit:** Computationally efficient and mitigates the vanishing gradient problem for positive values.")