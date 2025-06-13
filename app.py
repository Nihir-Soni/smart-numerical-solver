import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Numerical Methods
# -------------------------

# 1. Newton-Raphson Method
def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    steps = [x0]
    for _ in range(max_iter):
        x1 = x0 - f(x0) / df(x0)
        steps.append(x1)
        if abs(x1 - x0) < tol:
            break
        x0 = x1
    return x1, steps

# 2. Lagrange Interpolation
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

# 3. Trapezoidal Rule
def trapezoidal(f, a, b, n):
    h = (b - a) / n
    result = (f(a) + f(b)) / 2
    for i in range(1, n):
        result += f(a + i * h)
    return h * result

# -------------------------
# Streamlit UI
# -------------------------

st.title("🔢 Smart Numerical Solver & Visualizer")
st.write("A Streamlit-based app to solve mathematical problems using Python!")

method = st.sidebar.selectbox("Choose Method", ["Newton-Raphson", "Lagrange Interpolation", "Trapezoidal Rule"])

# Newton-Raphson UI
if method == "Newton-Raphson":
    fx = st.text_input("Enter f(x)", "x**3 - x - 2")
    dfx = st.text_input("Enter f'(x)", "3*x**2 - 1")
    x0 = st.number_input("Initial guess", value=1.5)
    f = lambda x: eval(fx)
    df = lambda x: eval(dfx)

    if st.button("Solve"):
        root, steps = newton_raphson(f, df, x0)
        st.success(f"Root found: {root:.6f}")
        st.line_chart(steps)

# Lagrange Interpolation UI
elif method == "Lagrange Interpolation":
    x_vals = st.text_input("x values (comma-separated)", "1,2,3")
    y_vals = st.text_input("y values (comma-separated)", "2,3,5")
    xp = st.number_input("Interpolate at x =", value=2.5)

    x = list(map(float, x_vals.split(',')))
    y = list(map(float, y_vals.split(',')))
    yp = lagrange_interpolation(x, y, xp)

    st.success(f"Interpolated value at x = {xp}: {yp:.4f}")
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o-', label="Given Points")
    ax.plot(xp, yp, 'rx', label="Interpolated Point")
    ax.legend()
    st.pyplot(fig)

# Trapezoidal Rule UI
elif method == "Trapezoidal Rule":
    fx = st.text_input("Enter function f(x)", "x**2")
    a = st.number_input("Lower limit a", value=0.0)
    b = st.number_input("Upper limit b", value=1.0)
    n = st.number_input("Number of intervals", value=4, step=1)
    f = lambda x: eval(fx)

    if st.button("Integrate"):
        result = trapezoidal(f, a, b, int(n))
        st.success(f"Integral ≈ {result:.6f}")
        # Plotting
        x_vals = np.linspace(a, b, 100)
        y_vals = [f(x) for x in x_vals]
        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, label="f(x)")
        ax.fill_between(x_vals, y_vals, alpha=0.3)
        st.pyplot(fig)
