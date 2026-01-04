"""
Fourier Interpolation with Grid Extension - Phase 3
Interactive Web Interface with Full-Width Configuration

Phase 3 Features:
- Setup tab with full-width code editors
- Two-column layouts (Config | Preview)
- Minimal sidebar
- Clean, modular structure
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sympy as sp
from sympy import sympify, lambdify
import pandas as pd
import math

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="HighFIE Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* Rich cream paper texture background - matching webpage style */
    .stApp {
        background-color: #fff8e7;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(255,250,240,0.5) 0%, transparent 50%),
            radial-gradient(circle at 90% 80%, rgba(255,250,240,0.5) 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, rgba(250,245,230,0.2) 0%, transparent 70%),
            repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(240,230,210,0.03) 2px, rgba(240,230,210,0.03) 4px),
            repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(240,230,210,0.03) 2px, rgba(240,230,210,0.03) 4px);
        background-size: 100% 100%, 100% 100%, 100% 100%, 40px 40px, 40px 40px;
    }
    
    /* Completely hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Expand main content to full width */
    .main .block-container {
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 1rem;
        background-color: transparent;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_download_button(fig, filename, label="📥 Download Plot (PNG, 300 DPI)", key=None):
    """Create a download button for a matplotlib figure.
    
    Args:
        fig: Matplotlib figure object
        filename: Name for the downloaded file (without extension)
        label: Button label
        key: Unique key for the button
    """
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    st.download_button(
        label=label,
        data=buf,
        file_name=f"{filename}.png",
        mime="image/png",
        key=key,
        use_container_width=True
    )

# =============================================================================
# FOURIER INTERPOLATION APP CLASS
# =============================================================================

class FourierInterpolationApp:
    def __init__(self):
        """Initialize the Fourier Interpolation application."""
        self.custom_extension_func = None
        self.custom_extension_params = {}
        
        # Load precomputed FD coefficients (symbolic, exact)
        # This avoids numerical instability from solving Vandermonde systems
        import json
        import os
        fd_table_path = os.path.join(os.path.dirname(__file__), 'fd_coefficients.json')
        try:
            with open(fd_table_path, 'r') as f:
                self.fd_table = json.load(f)
        except:
            # Fallback: if file not found, use empty table (will compute numerically)
            self.fd_table = {}
    
    def set_custom_extension(self, extension_func, params=None):
        """Set a custom extension function."""
        self.custom_extension_func = extension_func
        self.custom_extension_params = params or {}
    
    def fourier_eval(self, f_hat, x_eval):
        """Evaluate Fourier series at points x_eval."""
        n_extended = len(f_hat)
        result = np.zeros_like(x_eval, dtype=complex)
        
        for k in range(n_extended):
            result += f_hat[k] * np.exp(2j * np.pi * k * x_eval)
        
        return np.real(result)
    
    def fourier_eval_with_period(self, f_hat, x, xl, period):
        """Evaluate Fourier interpolant at points x with explicit period.
        
        Uses signed mode indexing for FFT coefficients.
        """
        n_coeffs = len(f_hat)
        result = np.zeros_like(x, dtype=complex)
        
        for k in range(n_coeffs):
            # FFT indexing: k -> signed mode
            sk = k if k <= n_coeffs // 2 else k - n_coeffs
            result += f_hat[k] * np.exp(2j * np.pi * sk * (x - xl) / period)
        
        return np.real(result)
    
    def compute_extension_and_fourier(self, f_vals, xl, xr, n, c, method, r):
        """Compute grid extension and Fourier coefficients."""
        extended = self.extend_grid_python(f_vals, xl, xr, c, method, r)
        coeffs = np.fft.fft(extended) / len(extended)
        return extended, coeffs
    
    def extend_grid_python(self, f, xl, xr, c, method, r):
        """Python implementation of grid extension."""
        f = np.array(f)
        n = len(f)
        
        # Check for custom extension
        if method == "Custom" and self.custom_extension_func is not None:
            return self.custom_extension_func(f, c, xl, xr, n, **self.custom_extension_params)
        
        if method == "Zero":
            return np.concatenate([f, np.zeros(c)])
        
        elif method == "Constant":
            return np.concatenate([f, np.full(c, f[-1])])
        
        elif method == "Periodic":
            if c <= n:
                return np.concatenate([f, f[:c]])
            else:
                num_full_periods = c // n
                remainder = c % n
                extension = np.tile(f, num_full_periods)
                if remainder > 0:
                    extension = np.concatenate([extension, f[:remainder]])
                return np.concatenate([f, extension])
        
        elif method == "Linear":
            slope = f[-1] - f[-2]
            extension = f[-1] + slope * np.arange(1, c + 1)
            return np.concatenate([f, extension])
        
        elif method == "Bump":
            def bump(t):
                t = np.clip(t, 0, 1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = np.exp(-1.0 / (t * (1.0 - t)))
                    result[t <= 0] = 0
                    result[t >= 1] = 0
                return result
            
            t = np.linspace(0, 1, c)
            weight = 1 - bump(t)
            extension = f[-1] * weight
            return np.concatenate([f, extension])
        
        elif method == "Hermite":
            # Proper Hermite extension - creates smooth periodic continuation
            return self.extend_hermite_proper(f, c, r)
        
        else:
            raise ValueError(f"Unknown extension method: {method}")
    
    def extend_hermite_proper(self, f, c, r):
        """
        Hermite extension matching derivatives at both boundaries.
        Creates a smooth periodic continuation.
        """
        n = len(f)
        h = 1.0 / n  # Normalized grid spacing
        xl = 0.0
        xr = 1.0
        a = xl
        
        # Compute derivative matrix at boundaries
        F = self.compute_fd_derivative_matrix(f, r, h, xl, xr, a)
        
        # Extend using Hermite interpolation
        extension = np.zeros(c)
        for j in range(c):
            x = a + (n + j) * h
            extension[j] = self.hermite_eval(x, F, r, h, xr, c)
        
        return np.concatenate([f, extension])
    
    def hermite_eval(self, x, F, r, h, xr, c):
        """
        Evaluate Hermite extension at point x.
        
        Uses two Hermite polynomials that blend together to create
        smooth periodic continuation.
        """
        per = c * h
        x1 = x - xr
        x2 = x - xr - per
        y1 = x1 / per
        y2 = -x2 / per
        
        p1 = 0.0
        p2 = 0.0
        factm = 1.0
        x1m = 1.0
        x2m = 1.0
        
        for m in range(r + 1):
            # Compute sum for this derivative order
            s1m = 0.0
            s2m = 0.0
            y1n = 1.0
            y2n = 1.0
            
            for n in range(r - m + 1):
                c_binom = self.binomial(r + n, r)
                s1m += c_binom * y1n
                s2m += c_binom * y2n
                y1n *= y1
                y2n *= y2
            
            # Add contribution from m-th derivative
            p1 += F[0][m] * x1m * s1m / factm
            p2 += F[1][m] * x2m * s2m / factm
            
            factm *= (m + 1)
            x1m *= x1
            x2m *= x2
        
        # Blend the two polynomials
        return (y2 ** (r + 1)) * p1 + (y1 ** (r + 1)) * p2
    
    def compute_fd_derivative_matrix(self, f, r, h, xl, xr, a):
        """
        Compute finite difference derivatives at boundaries.
        
        Returns F[0][m] = f^(m)(xr) and F[1][m] = f^(m)(xl)
        for m = 0, 1, ..., r
        """
        n = len(f)
        F = [[0.0 for _ in range(r + 1)] for _ in range(2)]
        
        if r == 0:
            F[0][0] = f[n - 1]
            F[1][0] = f[0]
            return F
        
        for m in range(r + 1):
            q = r
            # Coefficients for derivatives at left boundary
            c = self.fd_coefficients(m, q, (a - xl) / h)
            # Coefficients for derivatives at right boundary
            d = self.fd_coefficients(m, q, (xr - a - (n - 1) * h) / h)
            
            # Compute derivatives using finite differences
            sA = sum(f[j] * c[j] for j in range(min(m + q, n)))
            sB = sum(f[n - 1 - j] * d[j] for j in range(min(m + q, n)))
            
            F[1][m] = sA / (h ** m)
            F[0][m] = sB / ((-h) ** m)
        
        return F
    
    def fd_coefficients(self, m, q, a):
        """
        Compute finite difference coefficients for m-th derivative
        using q+1 points, offset by a.
        
        Uses precomputed symbolic values for m,q <= 10 to avoid 
        badly conditioned Vandermonde systems.
        """
        # Check if we have precomputed coefficients
        m_str, q_str, a_str = str(m), str(q), str(int(round(a)))
        
        if (m_str in self.fd_table and 
            q_str in self.fd_table[m_str] and 
            a_str in self.fd_table[m_str][q_str] and
            abs(a - int(round(a))) < 1e-10):
            # Use precomputed exact coefficients
            return np.array(self.fd_table[m_str][q_str][a_str], dtype=float)
        
        # Fall back to numerical computation (for unusual cases)
        # Use improved conditioning when possible
        N = m + q
        A = np.zeros((N, N))
        b = np.zeros(N)
        b[m] = 1.0
        
        # Build Vandermonde matrix
        for i in range(N):
            for j in range(N):
                A[i, j] = (a + j) ** i
        A[0, 0] = 1.0
        
        # Solve system with better conditioning
        try:
            # Try direct solve first
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            try:
                # Fall back to least squares with SVD
                c = np.linalg.lstsq(A, b, rcond=1e-12)[0]
            except:
                # Last resort: use pseudo-inverse
                c = np.linalg.pinv(A) @ b
        
        # Scale by factorial
        fact = math.factorial(m)
        c *= fact
        
        return c
    
    def binomial(self, n, k):
        """Compute binomial coefficient C(n, k)."""
        if k < 0 or k > n:
            return 0.0
        if k == 0 or k == n:
            return 1.0
        
        result = 1.0
        for i in range(1, k + 1):
            result *= (n - k + i) / i
        return result

# =============================================================================
# TAB FUNCTIONS
# =============================================================================

def setup_tab(app):
    """Setup tab with full-width configuration."""
    st.markdown("## ⚙️ Configure Your Analysis")
    st.markdown("Define your function, domain, extension method, and analysis parameters with full-width editors.")
    
    # Initialize session state for configuration
    if 'config' not in st.session_state:
        st.session_state.config = {
            'func_str': "sin(2*pi*x) * exp(-0.5*x)",
            'xl': 0.0,
            'xr': 1.0,
            'xl_str': '0',
            'xr_str': '1',
            'method': 'Hermite',
            'r': 4,
            'p': 1,
            'q': 1,
            'n_min': 8,
            'n_max': 64
        }
    
    # ==========================================================================
    # SECTION 1: FUNCTION DEFINITION
    # ==========================================================================
    st.markdown("---")
    st.subheader("1️⃣ Test Function")
    
    col_func_input, col_func_preview = st.columns([1, 1])
    
    with col_func_input:
        # Preset functions
        presets = {
            "sin(2πx)·exp(-0.5x)": "sin(2*pi*x) * exp(-0.5*x)",
            "Runge function": "1 / (1 + 25*x**2)",
            "Exponential decay": "exp(-5*x)",
            "Polynomial x³-x": "x**3 - x",
            "High frequency": "sin(10*pi*x)",
            "Gaussian": "exp(-50*(x-0.5)**2)",
            "Custom": None
        }
        
        preset_choice = st.selectbox(
            "Choose preset or custom",
            list(presets.keys()),
            index=0,
            help="Select a preset function or choose 'Custom' to define your own"
        )
        
        # Function input mode
        if preset_choice == "Custom":
            # Track previous mode to detect changes
            if 'prev_func_mode' not in st.session_state:
                st.session_state.prev_func_mode = "Simple Expression"
            
            func_mode = st.radio(
                "Input mode",
                ["Simple Expression", "Python Code"],
                horizontal=True,
                key="func_input_mode_radio"
            )
            
            # Detect mode change and reset function string ONLY if not loading
            if func_mode != st.session_state.prev_func_mode and 'loaded_func_code' not in st.session_state:
                st.session_state.prev_func_mode = func_mode
                # Reset to default for the new mode
                if func_mode == "Simple Expression":
                    st.session_state.config['func_str'] = "sin(2*pi*x) * exp(-0.5*x)"
                else:  # Python Code
                    st.session_state.config['func_str'] = """def f(x):
    x = np.atleast_1d(x)
    return np.sin(2*np.pi*x) * np.exp(-0.5*x)"""
            
            # Update previous mode
            st.session_state.prev_func_mode = func_mode
            
            if func_mode == "Simple Expression":
                with st.expander("ℹ️ Help: Simple Expressions"):
                    st.markdown("""
                    Use SymPy expressions with variable `x`:
                    - `sin(2*pi*x)` - Sine function
                    - `exp(-x)` - Exponential
                    - `x**2 + 1` - Polynomial
                    - `1/(1+x**2)` - Rational function
                    """)
                
                func_str = st.text_input(
                    "Function expression",
                    value=st.session_state.config['func_str'],
                    help="Enter a mathematical expression using x"
                )
                
                try:
                    x_sym = sp.Symbol('x')
                    expr = sympify(func_str)
                    func = lambdify(x_sym, expr, modules=['numpy'])
                    st.session_state.config['func_str'] = func_str
                except Exception as e:
                    st.error(f"Invalid expression: {e}")
                    return None, None, None
            
            else:  # Python Code
                with st.expander("📚 Help: Python Code"):
                    st.markdown("""
                    **Piecewise function:**
                    ```python
                    def f(x):
                        x = np.atleast_1d(x)
                        return np.where(x < 0, x**2, np.sin(x))
                    ```
                    
                    **Fourier series:**
                    ```python
                    def f(x):
                        result = 0
                        for n in range(1, 20):
                            result += np.sin(n*np.pi*x) / n**2
                        return result
                    ```
                    """)
                
                # Check if we should load code (from button or upload)
                if 'loaded_func_code' in st.session_state and st.session_state.loaded_func_code is not None:
                    default_func_code = st.session_state.loaded_func_code
                else:
                    default_func_code = """def f(x):
    x = np.atleast_1d(x)
    return np.sin(2*np.pi*x) * np.exp(-0.5*x)"""
                
                func_code = st.text_area(
                    "Python function definition",
                    value=default_func_code,
                    height=250,
                    help="Define function f(x) using numpy operations"
                )
                
                # Save/Load/Upload buttons inline
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    save_name = st.text_input("Name to save:", value="my_function", key="save_func_name", label_visibility="collapsed", placeholder="Name to save")
                
                with col2:
                    if st.button("💾 Save", key="save_func_btn", use_container_width=True):
                        if func_code and save_name:
                            st.session_state.saved_code_snippets[save_name] = {
                                'code': func_code,
                                'type': 'function'
                            }
                            st.success(f"✅ Saved '{save_name}'")
                
                with col3:
                    if st.session_state.saved_code_snippets:
                        func_snippets = {k: v for k, v in st.session_state.saved_code_snippets.items() if v['type'] == 'function'}
                        if func_snippets:
                            selected = st.selectbox("Load:", list(func_snippets.keys()), key="load_func_select", label_visibility="collapsed")
                            if st.button("📂 Load", key="load_func_btn", use_container_width=True):
                                st.session_state.loaded_func_code = func_snippets[selected]['code']
                                st.rerun()
                
                with col4:
                    uploaded = st.file_uploader("Upload", type=['py', 'txt'], key="upload_func_file", label_visibility="collapsed")
                    if uploaded:
                        # Check if this is a new upload using hash
                        file_hash = hash(uploaded.read())
                        uploaded.seek(0)  # Reset file pointer
                        
                        if 'last_uploaded_func_hash' not in st.session_state:
                            st.session_state.last_uploaded_func_hash = None
                        
                        if file_hash != st.session_state.last_uploaded_func_hash:
                            # New file uploaded
                            code = uploaded.read().decode('utf-8')
                            name = uploaded.name.replace('.py', '').replace('.txt', '')
                            st.session_state.saved_code_snippets[name] = {'code': code, 'type': 'function'}
                            st.session_state.loaded_func_code = code
                            st.session_state.last_uploaded_func_hash = file_hash
                            st.success(f"✅ Uploaded '{name}'!")
                            st.rerun()
                
                func_str = func_code
                st.session_state.config['func_str'] = func_str
                
                try:
                    namespace = {'np': np, 'numpy': np}
                    exec(func_code, namespace)
                    if 'f' not in namespace:
                        st.error("Must define a function named 'f'")
                        return None, None, None
                    func = namespace['f']
                except Exception as e:
                    st.error(f"Function error: {e}")
                    return None, None, None
        else:
            # Use preset
            func_str = presets[preset_choice]
            st.session_state.config['func_str'] = func_str
            x_sym = sp.Symbol('x')
            expr = sympify(func_str)
            func = lambdify(x_sym, expr, modules=['numpy'])
    
    # ==========================================================================
    # SECTION 2: DOMAIN
    # ==========================================================================
    st.markdown("---")
    st.subheader("2️⃣ Domain")
    
    col1, col2 = st.columns(2)
    with col1:
        xl_str = st.text_input(
            "Left boundary (x_l)", 
            value=str(st.session_state.config.get('xl_str', '0')),
            help="Enter a number or symbolic expression (e.g., 0, -pi, 1/3, -1)"
        )
        # Evaluate symbolic expression
        try:
            xl = float(sp.sympify(xl_str))
            st.session_state.config['xl'] = xl
            st.session_state.config['xl_str'] = xl_str
            st.caption(f"Evaluates to: {xl:.6f}")
        except Exception as e:
            st.error(f"Invalid expression: {e}")
            return None, None, None
            
    with col2:
        xr_str = st.text_input(
            "Right boundary (x_r)", 
            value=str(st.session_state.config.get('xr_str', '1')),
            help="Enter a number or symbolic expression (e.g., 1, pi, 2/3, sqrt(2))"
        )
        # Evaluate symbolic expression
        try:
            xr = float(sp.sympify(xr_str))
            st.session_state.config['xr'] = xr
            st.session_state.config['xr_str'] = xr_str
            st.caption(f"Evaluates to: {xr:.6f}")
        except Exception as e:
            st.error(f"Invalid expression: {e}")
            return None, None, None
    
    if xl >= xr:
        st.error("Left boundary must be less than right boundary")
        return None, None, None
    
    # Show function preview
    with col_func_preview:
        st.markdown("**Function Preview**")
        try:
            x_preview = np.linspace(xl, xr, 200)
            f_preview = func(x_preview)
            
            fig_preview, ax_preview = plt.subplots(figsize=(6, 4.5))
            ax_preview.plot(x_preview, f_preview, 'b-', linewidth=2)
            ax_preview.grid(True, alpha=0.3)
            ax_preview.set_xlabel('x', fontsize=10)
            ax_preview.set_ylabel('f(x)', fontsize=10)
            ax_preview.set_title(f'Function on [{xl:.2f}, {xr:.2f}]', fontsize=11, fontweight='bold')
            
            # Statistics
            f_min, f_max = np.min(f_preview), np.max(f_preview)
            f_mean = np.mean(f_preview)
            
            st.pyplot(fig_preview)
            create_download_button(fig_preview, "function_preview", key="dl_func_preview")
            plt.close(fig_preview)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Min", f"{f_min:.3f}")
            col2.metric("Max", f"{f_max:.3f}")
            col3.metric("Mean", f"{f_mean:.3f}")
            
        except Exception as e:
            st.error(f"Preview error: {e}")
    
    # ==========================================================================
    # SECTION 3: EXTENSION METHOD
    # ==========================================================================
    st.markdown("---")
    st.subheader("3️⃣ Extension Method")
    
    col_ext_config, col_ext_preview = st.columns([1, 1])
    
    with col_ext_config:
        # Extension mode selection
        extension_mode = st.radio(
            "Extension Mode",
            ["No Extension", "Built-in Methods", "Custom Code"],
            index=1,
            help="No Extension: c=0. Built-in: Predefined extensions. Custom Code: Define your own.",
            key="extension_mode_radio"
        )
        
        custom_extension_func = None
        custom_extension_params = {}
        
        if extension_mode == "No Extension":
            method = "Zero"
            r = 2
            st.session_state.config['method'] = method
            st.session_state.config['r'] = r
            st.info("No extension will be applied (c = 0)")
            # Set p and q to 0 for no extension
            p = 0
            q = 0
        
        elif extension_mode == "Built-in Methods":
            method = st.selectbox(
                "Method",
                ["Zero", "Constant", "Periodic", "Linear", "Hermite"],
                index=4
            )
            st.session_state.config['method'] = method
            
            # Hermite order
            r = 4
            if method == "Hermite":
                r = st.slider("Hermite order (r)", min_value=2, max_value=8, value=st.session_state.config['r'], step=1)
                st.session_state.config['r'] = r
        
        else:  # Custom Code
            st.markdown("**Define custom extension:**")
            
            # Show examples
            with st.expander("📚 Extension Examples"):
                st.markdown("""
                **Polynomial Extrapolation:**
                ```python
                def extend_custom(f, c, xl, xr, n, **params):
                    h = (xr - xl) / n
                    order = params.get('order', 2)
                    num_pts = min(order + 1, n)
                    x_fit = np.arange(n - num_pts, n) * h + xl
                    coeffs = np.polyfit(x_fit, f[-num_pts:], order)
                    x_ext = (n + np.arange(c)) * h + xl
                    extension = np.polyval(coeffs, x_ext)
                    return np.concatenate([f, extension])
                ```
                
                **Exponential Decay:**
                ```python
                def extend_custom(f, c, xl, xr, n, **params):
                    h = (xr - xl) / n
                    decay_rate = params.get('decay_rate', 2.0)
                    t = np.arange(1, c + 1) * h
                    extension = f[-1] * np.exp(-decay_rate * t)
                    return np.concatenate([f, extension])
                ```
                """)
            
            default_extension_code = """def extend_custom(f, c, xl, xr, n, **params):
    \"\"\"Custom extension method.\"\"\"
    h = (xr - xl) / n
    
    # Example: Linear extrapolation
    slope = (f[-1] - f[-2]) / h
    extension = f[-1] + slope * np.arange(1, c + 1) * h
    
    return np.concatenate([f, extension])"""
            
            # Check if we should load code
            if 'loaded_ext_code' in st.session_state and st.session_state.loaded_ext_code is not None:
                extension_code_value = st.session_state.loaded_ext_code
            else:
                extension_code_value = default_extension_code
            
            extension_code = st.text_area(
                "Extension code",
                value=extension_code_value,
                height=300,
                help="Define extend_custom(f, c, xl, xr, n, **params)"
            )
            
            # Save/Load/Upload buttons inline
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                save_ext_name = st.text_input("Name to save:", value="my_extension", key="save_ext_name", label_visibility="collapsed", placeholder="Name to save")
            
            with col2:
                if st.button("💾 Save", key="save_ext_btn", use_container_width=True):
                    if extension_code and save_ext_name:
                        st.session_state.saved_code_snippets[save_ext_name] = {
                            'code': extension_code,
                            'type': 'extension'
                        }
                        st.success(f"✅ Saved '{save_ext_name}'")
            
            with col3:
                if st.session_state.saved_code_snippets:
                    ext_snippets = {k: v for k, v in st.session_state.saved_code_snippets.items() if v['type'] == 'extension'}
                    if ext_snippets:
                        selected_ext = st.selectbox("Load:", list(ext_snippets.keys()), key="load_ext_select", label_visibility="collapsed")
                        if st.button("📂 Load", key="load_ext_btn", use_container_width=True):
                            st.session_state.loaded_ext_code = ext_snippets[selected_ext]['code']
                            st.rerun()
            
            with col4:
                uploaded_ext = st.file_uploader("Upload", type=['py', 'txt'], key="upload_ext_file", label_visibility="collapsed")
                if uploaded_ext:
                    # Check if this is a new upload using hash
                    ext_file_hash = hash(uploaded_ext.read())
                    uploaded_ext.seek(0)  # Reset file pointer
                    
                    if 'last_uploaded_ext_hash' not in st.session_state:
                        st.session_state.last_uploaded_ext_hash = None
                    
                    if ext_file_hash != st.session_state.last_uploaded_ext_hash:
                        # New file uploaded
                        ext_code = uploaded_ext.read().decode('utf-8')
                        ext_name = uploaded_ext.name.replace('.py', '').replace('.txt', '')
                        st.session_state.saved_code_snippets[ext_name] = {'code': ext_code, 'type': 'extension'}
                        st.session_state.loaded_ext_code = ext_code
                        st.session_state.last_uploaded_ext_hash = ext_file_hash
                        st.success(f"✅ Uploaded '{ext_name}'!")
                        st.rerun()
            
            # Optional parameters
            st.markdown("**Extension Parameters:**")
            num_params = st.number_input("Number of parameters", min_value=0, max_value=5, value=0, step=1)
            
            for i in range(num_params):
                col1, col2 = st.columns([2, 1])
                with col1:
                    param_name = st.text_input(f"Param {i+1} name", value=f"param{i+1}", key=f"pname_{i}")
                with col2:
                    param_value = st.number_input(f"Value", value=1.0, key=f"pval_{i}", format="%.3f")
                custom_extension_params[param_name] = param_value
            
            # Parse and validate
            try:
                namespace = {'np': np, 'numpy': np}
                exec(extension_code, namespace)
                
                if 'extend_custom' not in namespace:
                    st.error("Code must define 'extend_custom'")
                    return None, None, None
                
                custom_extension_func = namespace['extend_custom']
                app.set_custom_extension(custom_extension_func, custom_extension_params)
                method = "Custom"
                r = 4
                st.session_state.config['method'] = method
                
            except Exception as e:
                st.error(f"Code error: {e}")
                return None, None, None
    
    with col_ext_preview:
        st.markdown("**Extension Preview**")
        
        # Automatically show preview
        try:
            # Test data - use user-defined p and q
            n_test = 16
            if extension_mode == "No Extension":
                c_test = 0
            else:
                # Get p and q from config (they'll be set in Section 4)
                p_preview = st.session_state.config.get('p', 1)
                q_preview = st.session_state.config.get('q', 1)
                c_test = int((p_preview / q_preview) * n_test)
            
            x_test = np.linspace(xl, xr, n_test)
            f_test = func(x_test)
            
            # Run extension
            if extension_mode == "No Extension":
                # No extension, just use the original data
                extended_test = f_test
                st.caption(f"ℹ️ No extension (c = 0)")
            elif extension_mode == "Built-in Methods":
                extended_test = app.extend_grid_python(f_test, xl, xr, c_test, method, r)
                st.caption(f"ℹ️ Preview with n={n_test}, c={c_test} (using p={p_preview}, q={q_preview})")
            else:
                if custom_extension_func is None:
                    st.info("👆 Define custom extension code above to see preview")
                    return None, None, None
                extended_test = custom_extension_func(f_test, c_test, xl, xr, n_test, **custom_extension_params)
                st.caption(f"ℹ️ Preview with n={n_test}, c={c_test} (using p={p_preview}, q={q_preview})")
            
            # Validate
            if not isinstance(extended_test, np.ndarray):
                st.error("❌ Must return numpy array")
            elif len(extended_test) != n_test + c_test:
                st.error(f"❌ Wrong length: expected {n_test + c_test}, got {len(extended_test)}")
            elif c_test > 0 and not np.allclose(extended_test[:n_test], f_test, rtol=1e-10):
                st.error("❌ First n elements must equal input")
            elif not np.all(np.isfinite(extended_test)):
                st.error("❌ Contains NaN or Inf")
            else:
                # Bilateral preview
                fig_test, ax_test = plt.subplots(figsize=(7, 4))
                h_test = (xr - xl) / n_test
                
                if c_test == 0:
                    # No extension - just plot original grid
                    x_grid = xl + np.arange(n_test) * h_test
                    ax_test.plot(x_grid, extended_test, 'bo-', markersize=6, linewidth=1.5,
                               label='Original grid (no extension)', zorder=5)
                    ax_test.axvline(xl, color='gray', linestyle='--', alpha=0.6, linewidth=2,
                                  label='Domain boundaries')
                    ax_test.axvline(xr, color='gray', linestyle='--', alpha=0.6, linewidth=2)
                else:
                    # With extension
                    x_ext_grid = xl + np.arange(n_test + c_test) * h_test
                    x_left_ext = xl - np.arange(c_test, 0, -1) * h_test
                    f_left_ext = extended_test[n_test:]
                    
                    x_full = np.concatenate([x_left_ext, x_ext_grid])
                    f_full = np.concatenate([f_left_ext, extended_test])
                    
                    ax_test.plot(x_full, f_full, 'o-', markersize=4, linewidth=1.5, 
                               color='#2ca02c', alpha=0.7, label='Extended grid')
                    ax_test.axvspan(x_left_ext[0], xl, alpha=0.3, color='yellow', label='Extension region')
                    ax_test.axvspan(xr, x_ext_grid[-1], alpha=0.3, color='yellow')
                    
                    x_orig = xl + np.arange(n_test) * h_test
                    ax_test.plot(x_orig, extended_test[:n_test], 'bo', markersize=6, 
                               label='Original grid', zorder=5)
                    ax_test.plot(x_left_ext, f_left_ext, 'rs', markersize=5, 
                               label='Extension points', zorder=5)
                    ax_test.plot(x_ext_grid[n_test:], extended_test[n_test:], 'rs', markersize=5, zorder=5)
                    
                    ax_test.axvline(xl, color='gray', linestyle='--', alpha=0.6, linewidth=2, 
                                  label='Domain boundaries')
                ax_test.axvline(xr, color='gray', linestyle='--', alpha=0.6, linewidth=2)
                
                ax_test.legend(fontsize=8, ncol=2, loc='best')
                ax_test.grid(True, alpha=0.3)
                ax_test.set_title(f'Extension Preview: n={n_test}, c={c_test}', 
                                fontsize=10, fontweight='bold')
                ax_test.set_xlabel('x', fontsize=9)
                ax_test.set_ylabel('f(x)', fontsize=9)
                st.pyplot(fig_test)
                create_download_button(fig_test, "extension_preview", key="dl_ext_preview")
                plt.close(fig_test)
                
                st.caption(f"✅ Extended {n_test} → {n_test + c_test} points")
        
        except Exception as e:
            st.error(f"❌ Preview failed: {e}")
    
    # ==========================================================================
    # SECTION 4: GRID CONFIGURATION
    # ==========================================================================
    st.markdown("---")
    st.subheader("4️⃣ Grid Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if extension_mode == "No Extension":
            # Force p=0 for no extension and disable input
            p = 0
            st.number_input("Extension numerator (p)", min_value=0, max_value=0, 
                          value=0, step=1, disabled=True, 
                          help="Disabled: No extension mode uses p=0")
            st.session_state.config['p'] = 0
        else:
            # Ensure valid default when switching from No Extension
            default_p = max(1, st.session_state.config['p'])
            p = st.number_input("Extension numerator (p)", min_value=1, max_value=10, 
                              value=default_p, step=1)
            st.session_state.config['p'] = p
    
    with col2:
        if extension_mode == "No Extension":
            # Force q=1 for no extension and disable input
            q = 1
            st.number_input("Extension denominator (q)", min_value=1, max_value=1, 
                          value=1, step=1, disabled=True,
                          help="Disabled: No extension mode uses q=1")
            st.session_state.config['q'] = 1
        else:
            # Ensure valid default when switching from No Extension
            default_q = max(1, st.session_state.config['q'])
            q = st.number_input("Extension denominator (q)", min_value=1, max_value=10, 
                              value=default_q, step=1)
            st.session_state.config['q'] = q
    
    with col3:
        n_min = st.number_input("n min", min_value=4, max_value=128, 
                              value=st.session_state.config['n_min'], step=4)
        st.session_state.config['n_min'] = n_min
    
    with col4:
        n_max = st.number_input("n max", min_value=8, max_value=1024, 
                              value=st.session_state.config['n_max'], step=8)
        st.session_state.config['n_max'] = n_max
    
    if n_min >= n_max:
        st.error("n_min must be < n_max")
        return None, None, None
    
    n_levels = int(np.log2(n_max / n_min)) + 1
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Extension size: c = ({p}/{q}) × n")
    with col2:
        grid_sizes = [n_min * 2**i for i in range(n_levels)]
        st.info(f"Will test {n_levels} grids: {', '.join(map(str, grid_sizes[:5]))}{'...' if n_levels > 5 else ''}")
    
    # ==========================================================================
    # RUN BUTTON
    # ==========================================================================
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_analysis = st.button(
            "📈 Run Complete Analysis",
            type="primary",
            use_container_width=True,
            help="Execute analysis with current configuration",
            key="run_analysis_btn"
        )
    
    if run_analysis:
        with st.spinner("Running convergence analysis..."):
            results_list = []
            
            for level in range(n_levels):
                n_level = n_min * (2 ** level)
                
                if n_level % q != 0:
                    st.warning(f"Skipping n={n_level} (q={q} does not divide)")
                    continue
                
                c_level = (p * n_level) // q
                
                # Generate grid
                h = (xr - xl) / n_level
                x_grid = xl + np.arange(n_level) * h
                f_vals = func(x_grid)
                
                # Extend and compute Fourier
                extended, coeffs = app.compute_extension_and_fourier(
                    f_vals, xl, xr, n_level, c_level, method, r
                )
                
                extended_period = (xr - xl) * (1 + c_level / n_level)
                
                # Evaluate on fine grid
                n_fine = 2 * n_max
                x_fine = np.linspace(xl, xr, n_fine)
                f_true = func(x_fine)
                f_approx = app.fourier_eval_with_period(coeffs, x_fine, xl, extended_period)
                
                # Errors
                abs_error = np.abs(f_true - f_approx)
                max_abs_error = np.max(abs_error)
                
                max_f_orig = np.max(np.abs(f_vals))
                max_f_extended = np.max(np.abs(extended))
                
                max_rel_error = max_abs_error / max_f_orig if max_f_orig > 1e-15 else 0.0
                max_rel_error_extended = max_abs_error / max_f_extended if max_f_extended > 1e-15 else 0.0
                
                results_list.append({
                    'n': n_level,
                    'c': c_level,
                    'h': h,
                    'x_grid': x_grid,
                    'f_vals': f_vals,
                    'extended': extended,
                    'coeffs': coeffs,
                    'x_fine': x_fine,
                    'f_true': f_true,
                    'f_approx': f_approx,
                    'abs_error': abs_error,
                    'max_abs_error': max_abs_error,
                    'max_rel_error': max_rel_error,
                    'max_rel_error_extended': max_rel_error_extended,
                    'max_f_orig': max_f_orig,
                    'max_f_extended': max_f_extended,
                    'extended_period': extended_period
                })
            
            # Store results in BOTH for compatibility
            st.session_state.results = results_list
            st.session_state.results_list = results_list  # For Quick Test section
            st.session_state.analysis_params = {
                'xl': xl,
                'xr': xr,
                'func_str': func_str,
                'method': method,
                'extension_mode': extension_mode,
                'custom_extension_params': custom_extension_params if method == "Custom" else {},
                'r': r,
                'p': p,
                'q': q,
                'n_min': n_min,
                'n_max': n_max,
                'n_levels': n_levels
            }
            
            if not results_list:
                st.error("No valid grid sizes found! Adjust p and q.")
            else:
                st.success(f"✅ Analysis complete! {len(results_list)} grids tested. Scroll down to see results.")
    
    return func, func_str, method


def compare_tab():
    """Compare tab - add methods to compare against the Setup configuration."""
    
    st.markdown("### ⚖️ Compare Extension Methods")
    
    # Check if analysis has been run
    if not st.session_state.results_list or not st.session_state.analysis_params:
        st.warning("⚠️ Please run analysis in the **Setup & Test** tab first!")
        st.markdown("""
        **To use comparison:**
        1. Go to **Setup & Test** tab
        2. Configure your function, domain, and grid settings
        3. Choose a baseline extension method
        4. Click **Run Complete Analysis**
        5. Come back here to add more methods for comparison
        """)
        return
    
    # Get baseline configuration from Setup
    params = st.session_state.analysis_params
    baseline_results = st.session_state.results_list
    
    st.success("✅ Using configuration from Setup & Test tab")
    
    # Display baseline configuration (read-only)
    st.markdown("#### 📋 Baseline Configuration (from Setup & Test)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        func_display = params['func_str'][:50] + "..." if len(params['func_str']) > 50 else params['func_str']
        st.info(f"**Function**: `{func_display}`")
    
    with col2:
        st.info(f"**Domain**: [{params['xl']}, {params['xr']}]")
    
    with col3:
        st.info(f"**Grids**: {params['n_min']} to {params['n_max']}")
    
    # Show baseline method
    baseline_method = params['method']
    if baseline_method == 'Hermite':
        baseline_label = f"Hermite r={params['r']}"
    elif baseline_method == 'Custom':
        baseline_label = "Custom Extension"
    else:
        baseline_label = baseline_method
    
    st.info(f"**Baseline Method**: {baseline_label} | **Extension**: c = ({params['p']}/{params['q']}) × n")
    
    st.markdown("---")
    
    # Initialize comparison state
    if 'comparison_methods' not in st.session_state:
        st.session_state.comparison_methods = []
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {}
    if 'last_baseline_config' not in st.session_state:
        st.session_state.last_baseline_config = None
    
    # Create a hashable representation of current baseline config
    current_baseline_config = (
        baseline_label,
        params['func_str'],
        params['xl'],
        params['xr'],
        params['n_min'],
        params['n_max'],
        params['p'],
        params['q']
    )
    
    # Check if baseline configuration has changed - if so, reset comparison
    if st.session_state.last_baseline_config != current_baseline_config:
        st.session_state.comparison_results = {}
        st.session_state.comparison_methods = []
        st.session_state.last_baseline_config = current_baseline_config
        if st.session_state.last_baseline_config is not None:  # Don't show on first load
            st.info(f"🔄 Reset comparison for new configuration: {baseline_label}")
    
    # Add baseline to comparison results if not already there
    if baseline_label not in st.session_state.comparison_results:
        st.session_state.comparison_results[baseline_label] = {
            'config': params,
            'results': baseline_results
        }
    
    # ==========================================================================
    # ADD METHODS TO COMPARE
    # ==========================================================================
    
    st.markdown("#### ➕ Add Methods to Compare")
    st.info("💡 **Tip**: Select additional methods to compare against your baseline. All methods will use the same function, domain, and grid settings.")
    
    # Method selection
    available_methods = {
        "No Extension": {"method": "Zero", "r": 0, "p": 0, "q": 1},  # Special case
        "Zero Padding": {"method": "Zero", "r": 0},
        "Constant Extension": {"method": "Constant", "r": 0},
        "Periodic": {"method": "Periodic", "r": 0},
        "Linear": {"method": "Linear", "r": 0},
        "Hermite r=2": {"method": "Hermite", "r": 2},
        "Hermite r=4": {"method": "Hermite", "r": 4},
        "Hermite r=6": {"method": "Hermite", "r": 6},
    }
    
    # Remove baseline from available methods
    available_methods = {k: v for k, v in available_methods.items() if k != baseline_label}
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_methods = st.multiselect(
            "Select methods to add:",
            list(available_methods.keys()),
            default=[],
            help="Choose methods to compare against your baseline"
        )
    
    with col2:
        st.markdown("**Options**")
        same_pq = st.checkbox("Use same p/q", value=True, help="Use baseline p/q values")
        if not same_pq:
            # Handle case where baseline has p=0 (No Extension)
            default_p = max(1, params['p'])  # Ensure at least 1 for input
            default_q = max(1, params['q'])
            custom_p = st.number_input("p", 1, 10, default_p, key="compare_p")
            custom_q = st.number_input("q", 1, 10, default_q, key="compare_q")
        else:
            custom_p = params['p']
            custom_q = params['q']
    
    # Build configurations
    methods_to_run = []
    for method_name in selected_methods:
        if method_name not in st.session_state.comparison_results:
            config = available_methods[method_name].copy()
            # Handle No Extension special case
            if method_name == "No Extension":
                config['p'] = 0
                config['q'] = 1
            else:
                config['p'] = custom_p
                config['q'] = custom_q
            config['label'] = method_name
            methods_to_run.append(config)
    
    # ==========================================================================
    # RUN COMPARISON
    # ==========================================================================
    
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        run_comparison = st.button(
            f"🚀 Add {len(methods_to_run)} Method(s) to Comparison" if methods_to_run else "🚀 Run Comparison",
            type="primary",
            disabled=len(methods_to_run) == 0,
            use_container_width=True,
            key="compare_run_btn"
        )
    
    with col2:
        if st.button("🗑️ Clear All Comparisons", use_container_width=True, key="compare_clear_btn"):
            st.session_state.comparison_results = {
                baseline_label: {
                    'config': params,
                    'results': baseline_results
                }
            }
            st.rerun()
    
    # Run new methods
    if run_comparison and methods_to_run:
        # Parse function - handle both expressions and Python code
        try:
            func_str = params['func_str']
            
            # Check if it's Python code (contains 'def f(x):')
            if 'def f(x):' in func_str or 'def f(x)' in func_str:
                # Python code - execute it
                namespace = {'np': np, 'numpy': np}
                exec(func_str, namespace)
                if 'f' not in namespace:
                    st.error("Function code must define 'f'")
                    return
                func = namespace['f']
            else:
                # Expression - use sympify
                x_sym = sp.Symbol('x')
                expr = sympify(func_str)
                func = lambdify(x_sym, expr, modules=['numpy'])
        except Exception as e:
            st.error(f"Invalid function: {e}")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        app = st.session_state.app
        
        # Run each new method
        for idx, config in enumerate(methods_to_run):
            status_text.text(f"Running {config['label']}... ({idx+1}/{len(methods_to_run)})")
            
            results_list = []
            n = params['n_min']
            
            while n <= params['n_max']:
                # Validate q divides n
                if n % config['q'] != 0:
                    n *= 2
                    continue
                
                # Compute c
                c = (config['p'] * n) // config['q']
                
                # Compute grid
                h = (params['xr'] - params['xl']) / n
                x_grid = np.linspace(params['xl'], params['xr'], n + 1)
                f_vals = func(x_grid[:-1])
                
                # Extend and compute Fourier coefficients
                try:
                    extended, f_hat = app.compute_extension_and_fourier(
                        f_vals, params['xl'], params['xr'], n, c, config['method'], config['r']
                    )
                except Exception as e:
                    st.error(f"Error in {config['label']}: {e}")
                    break
                
                # Evaluate on fine grid
                n_fine = 2 * params['n_max']
                x_fine = np.linspace(params['xl'], params['xr'], n_fine)
                f_true = func(x_fine)
                
                period = (params['xr'] - params['xl']) * (1 + c / n)
                f_approx = app.fourier_eval_with_period(f_hat, x_fine, params['xl'], period)
                
                # Compute errors
                abs_error = np.abs(f_approx - f_true)
                max_abs_error = np.max(abs_error)
                
                max_f_orig = np.max(np.abs(f_vals))
                max_rel_error = max_abs_error / max_f_orig if max_f_orig > 0 else 0
                
                max_f_extended = np.max(np.abs(extended))
                max_rel_error_extended = max_abs_error / max_f_extended if max_f_extended > 0 else 0
                
                results_list.append({
                    'n': n,
                    'c': c,
                    'h': h,
                    'max_abs_error': max_abs_error,
                    'max_rel_error': max_rel_error,
                    'max_rel_error_extended': max_rel_error_extended,
                    'x_fine': x_fine,
                    'f_true': f_true,
                    'f_approx': f_approx,
                    'abs_error': abs_error,
                })
                
                n *= 2
            
            # Store results
            st.session_state.comparison_results[config['label']] = {
                'config': config,
                'results': results_list
            }
            
            progress_bar.progress((idx + 1) / len(methods_to_run))
        
        status_text.text("✅ Comparison complete!")
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Added {len(methods_to_run)} method(s) to comparison!")
        st.rerun()
    
    # ==========================================================================
    # DISPLAY COMPARISON RESULTS
    # ==========================================================================
    
    if len(st.session_state.comparison_results) > 1:  # More than just baseline
        st.markdown("---")
        st.subheader("📊 Comparison Results")
        
        st.info(f"Comparing {len(st.session_state.comparison_results)} method(s) including baseline")
        
        # Build comparison table
        comparison_data = []
        for label, data in st.session_state.comparison_results.items():
            results = data['results']
            finest = results[-1]
            config = data['config']
            
            # Compute average rate
            rates = []
            for i in range(len(results) - 1):
                if results[i]['max_rel_error'] > 0 and results[i+1]['max_rel_error'] > 0:
                    rate = np.log(results[i]['max_rel_error'] / results[i+1]['max_rel_error']) / np.log(2)
                    if np.isfinite(rate):
                        rates.append(rate)
            avg_rate = np.mean(rates) if rates else 0
            
            is_baseline = (label == baseline_label)
            
            comparison_data.append({
                'Method': label + (' ⭐' if is_baseline else ''),
                'Extension (c)': finest['c'],
                'Best Grid': finest['n'],
                'Max Abs Error': finest['max_abs_error'],
                'Max Rel Error': finest['max_rel_error'],
                'Avg Rate': avg_rate,
                'Quality': ('🌟' if finest['max_rel_error'] < 1e-10 else
                           '✅' if finest['max_rel_error'] < 1e-6 else
                           '⚠️' if finest['max_rel_error'] < 1e-3 else '❌')
            })
        
        # Create DataFrame and highlight winner
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        
        # Find winner (lowest error)
        winner_idx = df['Max Rel Error'].idxmin()
        
        # Display table with styling
        def highlight_rows(row):
            """Highlight winner and baseline rows."""
            # row.name is the index (row number)
            is_winner = (row.name == winner_idx)
            is_baseline_row = '⭐' in str(row['Method'])
            
            if is_winner:
                return ['background-color: #d4edda'] * len(row)  # Green for winner
            elif is_baseline_row:
                return ['background-color: #fff3cd'] * len(row)  # Yellow for baseline
            else:
                return [''] * len(row)
        
        st.dataframe(
            df.style.apply(highlight_rows, axis=1).format({
                'Max Abs Error': '{:.2e}',
                'Max Rel Error': '{:.2e}',
                'Avg Rate': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown(f"🏆 **Winner**: {df.loc[winner_idx, 'Method']} | ⭐ = Baseline from Setup")
        
        # Detailed visualizations
        st.markdown("---")
        st.markdown("#### 📈 Comparison Plots")
        
        # Plot 1: Convergence comparison
        fig_conv = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 2, figure=fig_conv, wspace=0.3)
        
        # Left: Convergence plot
        ax1 = fig_conv.add_subplot(gs[0, 0])
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(st.session_state.comparison_results)))
        
        for idx, (label, data) in enumerate(st.session_state.comparison_results.items()):
            results = data['results']
            grid_sizes = [r['n'] for r in results]
            rel_errors = [r['max_rel_error'] for r in results]
            
            # Highlight baseline
            linewidth = 3 if label == baseline_label else 2
            alpha = 1.0 if label == baseline_label else 0.7
            marker = 's' if label == baseline_label else 'o'
            
            ax1.loglog(grid_sizes, rel_errors, marker=marker, linestyle='-', 
                      color=colors[idx], linewidth=linewidth, markersize=8, 
                      label=label + (' (baseline)' if label == baseline_label else ''), 
                      alpha=alpha)
        
        ax1.set_xlabel('Grid size (n)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Max relative error', fontsize=13, fontweight='bold')
        ax1.set_title('Convergence Comparison', fontsize=15, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Right: Rate comparison
        ax2 = fig_conv.add_subplot(gs[0, 1])
        
        labels_list = []
        avg_rates = []
        colors_list = []
        
        for idx, (label, data) in enumerate(st.session_state.comparison_results.items()):
            results = data['results']
            rates = []
            for i in range(len(results) - 1):
                if results[i]['max_rel_error'] > 0 and results[i+1]['max_rel_error'] > 0:
                    rate = np.log(results[i]['max_rel_error'] / results[i+1]['max_rel_error']) / np.log(2)
                    if np.isfinite(rate):
                        rates.append(rate)
            
            labels_list.append(label + (' ⭐' if label == baseline_label else ''))
            avg_rates.append(np.mean(rates) if rates else 0)
            colors_list.append(colors[idx])
        
        bars = ax2.barh(labels_list, avg_rates, color=colors_list, alpha=0.7)
        ax2.set_xlabel('Average convergence rate', fontsize=13, fontweight='bold')
        ax2.set_title('Convergence Rate Comparison', fontsize=15, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Highlight best rate
        best_rate_idx = np.argmax(avg_rates)
        bars[best_rate_idx].set_alpha(1.0)
        bars[best_rate_idx].set_edgecolor('gold')
        bars[best_rate_idx].set_linewidth(3)
        
        st.pyplot(fig_conv)
        create_download_button(fig_conv, "convergence_comparison", key="dl_conv_comp")
        plt.close(fig_conv)
        
        # Plot 2: Error distribution comparison
        st.markdown("---")
        st.markdown("#### 🎯 Error Distribution (Finest Grid)")
        
        # Get finest grid from first method
        first_label = list(st.session_state.comparison_results.keys())[0]
        x_fine = st.session_state.comparison_results[first_label]['results'][-1]['x_fine']
        
        fig_error = plt.figure(figsize=(18, 5))
        gs_error = GridSpec(1, 3, figure=fig_error, wspace=0.3)
        
        # Absolute error
        ax1 = fig_error.add_subplot(gs_error[0, 0])
        for idx, (label, data) in enumerate(st.session_state.comparison_results.items()):
            abs_err = data['results'][-1]['abs_error']
            linewidth = 3 if label == baseline_label else 2
            alpha = 1.0 if label == baseline_label else 0.6
            ax1.plot(x_fine, abs_err, '-', color=colors[idx], linewidth=linewidth, 
                    label=label, alpha=alpha)
        
        ax1.set_xlabel('x', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Absolute error', fontsize=11, fontweight='bold')
        ax1.set_title('Absolute Error Comparison', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(params['xl'], params['xr'])
        
        # Approximation comparison
        ax2 = fig_error.add_subplot(gs_error[0, 1])
        
        # Plot true function
        f_true = st.session_state.comparison_results[first_label]['results'][-1]['f_true']
        ax2.plot(x_fine, f_true, 'k-', linewidth=2.5, label='True function', alpha=0.8)
        
        # Plot approximations
        for idx, (label, data) in enumerate(st.session_state.comparison_results.items()):
            f_approx = data['results'][-1]['f_approx']
            linestyle = '-' if label == baseline_label else '--'
            linewidth = 2 if label == baseline_label else 1.5
            alpha = 0.8 if label == baseline_label else 0.5
            ax2.plot(x_fine, f_approx, linestyle, color=colors[idx], linewidth=linewidth, 
                    label=f'{label}', alpha=alpha)
        
        ax2.set_xlabel('x', fontsize=11, fontweight='bold')
        ax2.set_ylabel('f(x)', fontsize=11, fontweight='bold')
        ax2.set_title('Approximation Comparison', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(params['xl'], params['xr'])
        
        # Relative error (log scale)
        ax3 = fig_error.add_subplot(gs_error[0, 2])
        max_f = np.max(np.abs(f_true))
        
        for idx, (label, data) in enumerate(st.session_state.comparison_results.items()):
            abs_err = data['results'][-1]['abs_error']
            rel_err = abs_err / max_f if max_f > 0 else abs_err
            linewidth = 3 if label == baseline_label else 2
            alpha = 1.0 if label == baseline_label else 0.6
            ax3.semilogy(x_fine, rel_err, '-', color=colors[idx], linewidth=linewidth, 
                        label=label, alpha=alpha)
        
        ax3.set_xlabel('x', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Relative error', fontsize=11, fontweight='bold')
        ax3.set_title('Relative Error Comparison (log)', fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3, which='both')
        ax3.set_xlim(params['xl'], params['xr'])
        
        st.pyplot(fig_error)
        create_download_button(fig_error, "error_distribution_comparison", key="dl_error_comp")
        plt.close(fig_error)
        
        # Export comparison data
        st.markdown("---")
        st.markdown("#### 💾 Export Comparison Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as CSV
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison Table (CSV)",
                data=csv_data,
                file_name="fourier_comparison.csv",
                mime="text/csv"
            )
        
        with col2:
            st.info("💡 **Tip**: Add more methods above or clear all to start fresh.")
    
    else:
        st.info("👆 Add methods above to see comparison results. The baseline from Setup & Test is already included.")
# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Initialize app
    if 'app' not in st.session_state:
        st.session_state.app = FourierInterpolationApp()
    
    # Initialize session state variables
    if 'results_list' not in st.session_state:
        st.session_state.results_list = []
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'analysis_params' not in st.session_state:
        st.session_state.analysis_params = {}
    
    # Session storage for custom code
    if 'saved_code_snippets' not in st.session_state:
        st.session_state.saved_code_snippets = {}
    
    app = st.session_state.app
    
    # Header
    st.markdown('<h1 class="main-header">🔬 HighFIE Lab</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>High-Order Fourier Interpolation with Extension</b><br>
    An interface for developing and testing high-order Fourier Interpolation with grid extension
    </div>
    """, unsafe_allow_html=True)
    
    # Main tabs - SIMPLIFIED to 2 tabs
    tab_setup, tab_compare = st.tabs([
        "⚙️ Setup & Test",
        "⚖️ Compare"
    ])
    
    # ==========================================================================
    # TAB 1: SETUP & TEST
    # ==========================================================================
    with tab_setup:
        setup_tab(app)
        
        # Add quick test section at bottom
        st.markdown("---")
        st.markdown("### 🧪 Quick Test")
        st.info("💡 **Tip**: Test your configuration with a single method before running full comparison.")
        
        if st.session_state.results_list:
            st.markdown("#### Latest Test Results")
            
            # Quick summary metrics
            results = st.session_state.results_list
            finest = results[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Best Grid", f"n = {finest['n']}")
            
            with col2:
                st.metric("Max Abs Error", f"{finest['max_abs_error']:.2e}")
            
            with col3:
                st.metric("Max Rel Error", f"{finest['max_rel_error']:.2e}")
            
            with col4:
                # Compute average rate
                rates = []
                for i in range(len(results) - 1):
                    if results[i]['max_rel_error'] > 0 and results[i+1]['max_rel_error'] > 0:
                        rate = np.log(results[i]['max_rel_error'] / results[i+1]['max_rel_error']) / np.log(2)
                        if np.isfinite(rate):
                            rates.append(rate)
                avg_rate = np.mean(rates) if rates else 0
                st.metric("Avg Rate", f"{avg_rate:.2f}")
            
            # Grid selector
            st.markdown("#### Select Grid Size for Detailed Plots")
            selected_idx = st.selectbox(
                "Grid size:",
                range(len(results)),
                format_func=lambda i: f"n = {results[i]['n']}",
                index=len(results) - 1,
                help="Choose which grid size to visualize in detail",
                key="quick_test_grid_selector"
            )
            
            selected_result = results[selected_idx]
            
            # Show detailed plots
            st.markdown(f"#### Detailed Analysis (n = {selected_result['n']})")
            
            # Get parameters for plotting
            params = st.session_state.analysis_params
            xl = params['xl']
            xr = params['xr']
            
            # Create 3 rows of plots
            # Row 1: Extended grid and approximation (2 plots)
            fig1 = plt.figure(figsize=(18, 5))
            gs1 = GridSpec(1, 2, figure=fig1, wspace=0.25)
            
            # Plot 1: Extended grid function
            ax1 = fig1.add_subplot(gs1[0, 0])
            
            h = selected_result['h']
            n = selected_result['n']
            c = selected_result['c']
            x_grid = selected_result['x_grid']
            extended = selected_result['extended']
            
            if c == 0:
                # No extension - just plot original grid
                ax1.plot(x_grid, extended, 'bo-', markersize=6, linewidth=1.5,
                        label='Original grid (no extension)', zorder=5)
                ax1.axvline(xl, color='gray', linestyle='--', alpha=0.6, linewidth=2,
                           label='Domain boundaries')
                ax1.axvline(xr, color='gray', linestyle='--', alpha=0.6, linewidth=2)
            else:
                # Create full grid including left extension
                x_ext_grid = xl + np.arange(n + c) * h
                x_left_ext = xl - np.arange(c, 0, -1) * h
                f_left_ext = extended[n:]
                
                x_full = np.concatenate([x_left_ext, x_ext_grid])
                f_full = np.concatenate([f_left_ext, extended])
                
                ax1.plot(x_full, f_full, 'o-', markersize=4, linewidth=1.5, color='#2ca02c', 
                         alpha=0.7, label='Extended grid')
                ax1.axvspan(x_left_ext[0], xl, alpha=0.3, color='yellow', label='Extension region')
                ax1.axvspan(xr, x_ext_grid[-1], alpha=0.3, color='yellow')
                
                ax1.plot(x_grid, extended[:n], 'bo', markersize=6, label='Original grid', zorder=5)
                ax1.plot(x_left_ext, f_left_ext, 'rs', markersize=5, label='Extension points', zorder=5)
                ax1.plot(x_ext_grid[n:], extended[n:], 'rs', markersize=5, zorder=5)
                
                ax1.axvline(xl, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='Domain boundaries')
                ax1.axvline(xr, color='gray', linestyle='--', alpha=0.6, linewidth=2)
            
            ax1.set_xlabel('x', fontsize=11, fontweight='bold')
            ax1.set_ylabel('f(x)', fontsize=11, fontweight='bold')
            ax1.set_title(f'Extended Function Grid (n={n}, c={c})', fontsize=13, fontweight='bold')
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Function vs Approximation
            ax2 = fig1.add_subplot(gs1[0, 1])
            
            ax2.plot(selected_result['x_fine'], selected_result['f_true'], 'b-', linewidth=2, 
                    label='True function', alpha=0.7)
            ax2.plot(selected_result['x_fine'], selected_result['f_approx'], 'r--', linewidth=2, 
                    label='Fourier approximation', alpha=0.7)
            ax2.plot(x_grid, selected_result['f_vals'], 'go', markersize=5, label='Grid points', zorder=5)
            
            ax2.set_xlabel('x', fontsize=11, fontweight='bold')
            ax2.set_ylabel('f(x)', fontsize=11, fontweight='bold')
            ax2.set_title('True Function vs Fourier Approximation', fontsize=13, fontweight='bold')
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(xl, xr)
            
            plt.tight_layout()
            st.pyplot(fig1)
            create_download_button(fig1, f"extended_grid_approximation_n{selected_result['n']}", key="dl_fig1")
            plt.close(fig1)
            
            # Row 2: Error profiles (2 plots)
            fig2 = plt.figure(figsize=(18, 5))
            gs2 = GridSpec(1, 2, figure=fig2, wspace=0.25)
            
            # Plot 3: Absolute error
            ax3 = fig2.add_subplot(gs2[0, 0])
            ax3.plot(selected_result['x_fine'], selected_result['abs_error'], 'r-', linewidth=2, alpha=0.7)
            ax3.axhline(selected_result['max_abs_error'], color='k', linestyle='--', 
                       label=f"Max = {selected_result['max_abs_error']:.2e}")
            ax3.set_xlabel('x', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Absolute error', fontsize=11, fontweight='bold')
            ax3.set_title('Absolute Error: |f(x) - f̃(x)|', fontsize=13, fontweight='bold')
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(xl, xr)
            
            # Plot 4: Relative errors
            ax4 = fig2.add_subplot(gs2[0, 1])
            
            # Relative error (standard): abs_error / max|f| on original grid points
            max_f_orig = selected_result['max_f_orig']
            pointwise_rel = selected_result['abs_error'] / max_f_orig
            ax4.semilogy(selected_result['x_fine'], pointwise_rel, 'r-', linewidth=2, 
                        label=f'Rel error: ÷ max|f(x_i)| on orig grid = {max_f_orig:.3f}')
            
            # Relative error (extended): abs_error / max|f| on extended grid
            max_f_extended = selected_result['max_f_extended']
            pointwise_rel_ext = selected_result['abs_error'] / max_f_extended
            ax4.semilogy(selected_result['x_fine'], pointwise_rel_ext, 'g-', linewidth=2, 
                        alpha=0.7, label=f'Rel error (ext): ÷ max|extended(x_i)| = {max_f_extended:.3f}')
            
            # Add horizontal lines showing the max relative errors
            ax4.axhline(y=selected_result['max_rel_error'], color='r', linestyle='--', alpha=0.5, linewidth=1,
                       label=f'Max rel error = {selected_result["max_rel_error"]:.2e}')
            ax4.axhline(y=selected_result['max_rel_error_extended'], color='g', linestyle='--', alpha=0.5, linewidth=1,
                       label=f'Max rel error (ext) = {selected_result["max_rel_error_extended"]:.2e}')
            
            ax4.set_xlabel('x', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Relative Error', fontsize=11, fontweight='bold')
            ax4.set_title(f'Relative Approximation Errors (n={selected_result["n"]})', fontsize=13, fontweight='bold')
            ax4.legend(loc='best', fontsize=9)
            ax4.grid(True, alpha=0.3, which='both')
            ax4.set_xlim(xl, xr)
            
            plt.tight_layout()
            st.pyplot(fig2)
            create_download_button(fig2, f"error_profiles_n{selected_result['n']}", key="dl_fig2")
            plt.close(fig2)
            
            # Row 3: Convergence analysis (plot + table)
            st.markdown("#### Convergence Analysis")
            
            fig3 = plt.figure(figsize=(18, 10))
            gs3 = GridSpec(2, 1, figure=fig3, hspace=0.35)
            
            # Plot 5: Convergence
            ax5 = fig3.add_subplot(gs3[0, 0])
            grid_sizes = [r['n'] for r in results]
            rel_errors = [r['max_rel_error'] for r in results]
            rel_errors_ext = [r['max_rel_error_extended'] for r in results]
            
            ax5.loglog(grid_sizes, rel_errors, 'bo-', linewidth=2, markersize=8, 
                      label='Relative error')
            ax5.loglog(grid_sizes, rel_errors_ext, 'rs--', linewidth=2, markersize=8, 
                      label='Relative error (extended)')
            
            # Reference line
            if len(grid_sizes) >= 3:
                log_n = np.log(grid_sizes[-3:])
                log_err = np.log(rel_errors[-3:])
                if all(np.isfinite(log_err)):
                    slope = np.polyfit(log_n, log_err, 1)[0]
                    ax5.plot(grid_sizes, rel_errors[0] * (np.array(grid_sizes) / grid_sizes[0])**slope,
                            '--', color='gray', alpha=0.8, linewidth=2.5, label=f'Slope ≈ {slope:.2f}')
            
            ax5.set_xlabel('Grid size (n)', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Relative error', fontsize=14, fontweight='bold')
            ax5.set_title('Convergence of Fourier Interpolation', fontsize=16, fontweight='bold')
            ax5.legend(loc='best', fontsize=11)
            ax5.grid(True, alpha=0.3, which='both')
            
            # Table: Convergence data
            ax_table = fig3.add_subplot(gs3[1, 0])
            ax_table.axis('off')
            
            # Compute rates
            rates = []
            for i in range(len(rel_errors) - 1):
                if rel_errors[i] > 0 and rel_errors[i+1] > 0:
                    rate = np.log(rel_errors[i] / rel_errors[i+1]) / np.log(2)
                    rates.append(rate)
                else:
                    rates.append(np.nan)
            
            rates_ext = []
            for i in range(len(rel_errors_ext) - 1):
                if rel_errors_ext[i] > 0 and rel_errors_ext[i+1] > 0:
                    rate = np.log(rel_errors_ext[i] / rel_errors_ext[i+1]) / np.log(2)
                    rates_ext.append(rate)
                else:
                    rates_ext.append(np.nan)
            
            table_data = []
            headers = ['Grid Size (n)', 'Extension (c)', 'Max Abs Error', 'Rel Error', 'Rel Error (ext)', 'Rate', 'Rate (ext)']
            
            for i, res in enumerate(results):
                row = [
                    f"{res['n']}",
                    f"{res['c']}",
                    f"{res['max_abs_error']:.2e}",
                    f"{res['max_rel_error']:.2e}",
                    f"{res['max_rel_error_extended']:.2e}",
                    f"{rates[i]:.2f}" if i < len(rates) and not np.isnan(rates[i]) else "—",
                    f"{rates_ext[i]:.2f}" if i < len(rates_ext) and not np.isnan(rates_ext[i]) else "—"
                ]
                table_data.append(row)
            
            table = ax_table.table(cellText=table_data, colLabels=headers,
                                 cellLoc='center', loc='center',
                                 colWidths=[0.13, 0.13, 0.15, 0.15, 0.15, 0.11, 0.11])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2.5)
            
            # Style header
            for i in range(len(headers)):
                cell = table[(0, i)]
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(len(table_data)):
                for j in range(len(headers)):
                    cell = table[(i+1, j)]
                    if i % 2 == 0:
                        cell.set_facecolor('#E7E6E6')
            
            ax_table.set_title('Convergence Analysis Results', fontsize=15, fontweight='bold', pad=30)
            
            plt.tight_layout()
            st.pyplot(fig3)
            create_download_button(fig3, "convergence_analysis", key="dl_fig3")
            
            # Download buttons for convergence analysis
            st.markdown("#### 📥 Export Convergence Data")
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                # Export table as CSV
                import io
                csv_buffer = io.StringIO()
                csv_buffer.write("Grid_Size_n,Extension_c,Max_Abs_Error,Rel_Error,Rel_Error_ext,Rate,Rate_ext\n")
                
                for i, res in enumerate(results):
                    rate_str = f"{rates[i]:.6f}" if i < len(rates) and not np.isnan(rates[i]) else ""
                    rate_ext_str = f"{rates_ext[i]:.6f}" if i < len(rates_ext) and not np.isnan(rates_ext[i]) else ""
                    
                    csv_buffer.write(f"{res['n']},{res['c']},{res['max_abs_error']:.6e}," +
                                    f"{res['max_rel_error']:.6e},{res['max_rel_error_extended']:.6e}," +
                                    f"{rate_str},{rate_ext_str}\n")
                
                st.download_button(
                    label="📥 Download Table (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="convergence_table.csv",
                    mime="text/csv",
                    key="download_convergence_csv",
                    use_container_width=True
                )
            
            with col_dl2:
                # Export plot as PNG
                import io as io_png
                buf = io_png.BytesIO()
                fig3.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                st.download_button(
                    label="📥 Download Plot (PNG, 300 DPI)",
                    data=buf,
                    file_name="convergence_plot.png",
                    mime="image/png",
                    key="download_convergence_plot",
                    use_container_width=True
                )
            
            plt.close(fig3)
            
            st.success(f"✅ Configuration tested successfully! Ready for comparison.")
            
            # ==================================================================
            # ==================================================================
            # POST-PROCESSING SECTION - DISABLED FOR V1
            # ==================================================================
            # Will be enabled in a future version
            
        else:
            st.info("👆 Click **Run Analysis** above to test your configuration before comparing.")
    
    # TAB 2: COMPARE
    # ==========================================================================
    with tab_compare:
        compare_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <small>
    <b>HighFIE Lab</b> - High-Order Fourier Interpolation with Extension<br>
    Developed at <b>Indian Institute of Technology Kanpur</b><br>
    Built with Streamlit | Based on spectral methods for function approximation
    </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
