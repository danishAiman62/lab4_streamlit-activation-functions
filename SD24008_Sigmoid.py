import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sigmoid Activation Visualiser", layout="centered")

st.title("📈 Sigmoid Activation Function Visualiser")
st.markdown("**Student ID:** SD24008 | **Subject:** BSD3513 Introduction to AI")

st.write("""
The **Sigmoid** activation function maps any real-valued number into a range between 0 and 1. 
It is frequently used in the output layer for binary classification problems.
""")

# Controls
x_range = st.slider("Select Input Range (x)", -10.0, 10.0, (-6.0, 6.0), 0.5)
show_derivative = st.checkbox("Show Derivative / Gradient", value=True)

# Generate data
x = np.linspace(x_range[0], x_range[1], 400)
s = 1 / (1 + np.exp(-x))
ds = s * (1 - s)

# Plotting
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(x, s, label="Sigmoid: f(x) = 1 / (1 + e^-x)", color="#1b9e77", linewidth=3)
if show_derivative:
    ax.plot(x, ds, label="Derivative: f'(x)", color="#e7298a", linestyle="--", linewidth=2)

ax.axhline(0, color='black', linewidth=1, linestyle='-')
ax.axvline(0, color='black', linewidth=1, linestyle='-')
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_title("Sigmoid Activation Function & Gradient", fontsize=12, fontweight='bold')
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.legend()
st.pyplot(fig)

st.subheader("Mathematical Properties")
st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")
st.latex(r"f'(x) = f(x)(1 - f(x))")
st.info("**Key Characteristic:** Outputs probabilities between 0 and 1, but susceptible to vanishing gradients for extreme input values.")