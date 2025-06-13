# Smart Numerical Solver - Enhanced Streamlit App

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Smart Numerical Solver", layout="wide")

# Header
st.markdown("""
    <h1 style='text-align: center; color: #00ffe7;'>🧮 Smart Numerical Solver & Visualizer</h1>
    <p style='text-align: center; color: #a0f0ff;'>Built with Python + Streamlit by Nihir Soni</p>
    <hr style='border: 1px solid #00ffe7;'>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Choose Numerical Method")
method = st.sidebar.radio("Select one:", ["Newton-Raphson", "Lagrange Interpolation", "Trapezoidal Rule"])

# Newton-Raphson Method
if method == "Newton-Raphson":
    st.subheader("🧠 Newton-Raphson Method")
    fx = st.text_input("Enter f(x):", "x**3 - x - 2")
    dfx = st.text_input("Enter f'(x):", "3*x**2 - 1")
    x0 = st.number_input("Initial guess (x0):", value=1.5)
    f = lambda x: eval(fx)
    df = lambda x: eval(dfx)

    def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
        steps = [x0]
        for _ in range(max_iter):
            x1 = x0 - f(x0)/df(x0)
            steps.append(x1)
            if abs(x1 - x0) < tol:
                break
            x0 = x1
        return x1, steps

    if st.button("Solve"):
        try:
            root, steps = newton_raphson(f, df, x0)
            st.success(f"✅ Root found: {root:.6f}")

            # Plotting
            fig, ax = plt.subplots()
            x_vals = np.linspace(x0-3, x0+3, 400)
            y_vals = [f(x) for x in x_vals]
            ax.plot(x_vals, y_vals, label="f(x)")
            ax.axhline(0, color='gray', linestyle='--')
            ax.plot(steps, [f(x) for x in steps], 'ro-', label='Iterations')
            ax.set_title("Newton-Raphson Iterations")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

# Lagrange Interpolation
elif method == "Lagrange Interpolation":
    st.subheader("📈 Lagrange Interpolation")
    x_vals = st.text_input("Enter x values (comma-separated):", "1,2,3")
    y_vals = st.text_input("Enter y values (comma-separated):", "2,3,5")
    xp = st.number_input("Interpolate at x =", value=2.5)

    def lagrange_interpolation(x, y, xp):
        yp = 0
        n = len(x)
        for i in range(n):
            p = 1
            for j in range(n):
                if i != j:
                    p *= (xp - x[j]) / (x[i] - x[j])
            yp += p * y[i]
        return yp

    if st.button("Interpolate"):
        try:
            x = list(map(float, x_vals.split(',')))
            y = list(map(float, y_vals.split(',')))
            yp = lagrange_interpolation(x, y, xp)
            st.success(f"🎯 Interpolated value at x = {xp}: {yp:.4f}")

            # Plotting
            fig, ax = plt.subplots()
            ax.plot(x, y, 'o-', label="Data Points")
            ax.plot(xp, yp, 'rx', label="Interpolated Point")
            ax.set_title("Lagrange Interpolation")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

# Trapezoidal Rule
elif method == "Trapezoidal Rule":
    st.subheader("📐 Trapezoidal Rule for Integration")
    fx = st.text_input("Enter function f(x):", "x**2")
    a = st.number_input("Lower limit (a):", value=0.0)
    b = st.number_input("Upper limit (b):", value=1.0)
    n = st.number_input("Number of intervals (n):", value=4, step=1)
    f = lambda x: eval(fx)

    def trapezoidal(f, a, b, n):
        h = (b - a) / n
        result = (f(a) + f(b)) / 2
        for i in range(1, int(n)):
            result += f(a + i * h)
        return h * result

    if st.button("Integrate"):
        try:
            result = trapezoidal(f, a, b, n)
            st.success(f"🧮 Integral ≈ {result:.6f}")

            # Plotting
            x_vals = np.linspace(a, b, 100)
            y_vals = [f(x) for x in x_vals]
            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals, label="f(x)")
            ax.fill_between(x_vals, y_vals, alpha=0.3, label="Area")
            ax.set_title("Trapezoidal Integration")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.markdown("""
    <hr style='border: 1px solid #00ffe7;'>
    <p style='text-align: center; color: #888;'>© 2025 Nihir Soni | Numerical Methods Visualizer</p>
""", unsafe_allow_html=True)
