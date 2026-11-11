import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tanh Activation Visualiser", layout="centered")

st.title("📉 Hyperbolic Tangent (Tanh) Activation Visualiser")
st.markdown("**Student ID:** SD24008 | **Subject:** BSD3513 Introduction to AI")

st.write("""
The **Tanh** (Hyperbolic Tangent) activation function squashes real-valued numbers to a range between -1 and 1. 
Because it is zero-centered, it generally performs better than the Sigmoid function in hidden layers.
""")

# Controls
x_range = st.slider("Select Input Range (x)", -10.0, 10.0, (-6.0, 6.0), 0.5)
show_derivative = st.checkbox("Show Derivative / Gradient", value=True)

# Generate data
x = np.linspace(x_range[0], x_range[1], 400)
t = np.tanh(x)
dt = 1.0 - np.tanh(x)**2

# Plotting
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(x, t, label="Tanh: f(x) = tanh(x)", color="#7570b3", linewidth=3)
if show_derivative:
    ax.plot(x, dt, label="Derivative: f'(x)", color="#e6ab02", linestyle="--", linewidth=2)

ax.axhline(0, color='black', linewidth=1, linestyle='-')
ax.axvline(0, color='black', linewidth=1, linestyle='-')
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_title("Hyperbolic Tangent (Tanh) & Gradient", fontsize=12, fontweight='bold')
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.legend()
st.pyplot(fig)

st.subheader("Mathematical Properties")
st.latex(r"f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}")
st.latex(r"f'(x) = 1 - \tanh^2(x)")
st.info("**Key Benefit:** Zero-centered output range $(-1, 1)$, making optimization easier across successive layers.")