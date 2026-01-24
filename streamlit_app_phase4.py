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
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* ========================================================================
       LIGHT MODE (Default) - Off-white background with clean styling
       ======================================================================== */
    
    /* Clean off-white background - matching academic webpage style */
    .stApp {
        background-color: #f9f9f9;
        background-image: none;
    }
    
    /* Make tabs more prominent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #e8e8e8;
        padding: 8px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 6px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e3a5f !important;
        color: white !important;
    }
    
    /* ========================================================================
       DARK MODE Support - Adapt colors for dark mode
       ======================================================================== */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #1a1a2e !important;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(26,26,46,0.8) 0%, transparent 50%),
                radial-gradient(circle at 90% 80%, rgba(26,26,46,0.8) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(20,20,40,0.4) 0%, transparent 70%),
                repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(30,30,50,0.1) 2px, rgba(30,30,50,0.1) 4px),
                repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(30,30,50,0.1) 2px, rgba(30,30,50,0.1) 4px);
        }
        
        /* Adjust text colors for dark mode */
        .main-header {
            color: #4da6ff !important;
        }
        
        /* Adjust info boxes for dark mode */
        .info-box {
            background-color: #16213e !important;
            border-left-color: #4da6ff !important;
        }
        
        .success-box {
            background-color: #1a3a1a !important;
            border-left-color: #4caf50 !important;
        }
        
        .warning-box {
            background-color: #3a2a1a !important;
            border-left-color: #ff9800 !important;
        }
        
        .error-box {
            background-color: #3a1a1a !important;
            border-left-color: #f44336 !important;
        }
        
        /* Input labels in dark mode */
        .stTextInput > label,
        .stNumberInput > label,
        .stSelectbox > label,
        .stMultiSelect > label,
        .stTextArea > label {
            color: #b0b0b0 !important;
        }
    }
    
    /* Completely hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Hide Streamlit footer and menu */
    footer {
        visibility: hidden;
    }
    #MainMenu {
        visibility: hidden;
    }
    header {
        visibility: hidden;
    }
    
    /* Expand main content to full width */
    .main .block-container {
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 1rem;
        background-color: transparent;
    }
    
    /* Dark theme for all input widgets (works in both light and dark mode) */
    /* Text inputs */
    input[type="text"],
    input[type="number"],
    textarea,
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        color: #e0e0e0 !important;
        border: 2px solid #0f3460 !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        font-weight: 500 !important;
        caret-color: #4da6ff !important;  /* Bright blue cursor for visibility */
    }
    
    /* Input focus state */
    input[type="text"]:focus,
    input[type="number"]:focus,
    textarea:focus,
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {
        border-color: #1f77b4 !important;
        box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2) !important;
    }
    
    /* Select boxes and multiselect */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        color: #e0e0e0 !important;
        border: 2px solid #0f3460 !important;
        border-radius: 8px !important;
    }
    
    /* Dropdown menus */
    [data-baseweb="popover"] {
        background: #16213e !important;
    }
    
    [role="option"] {
        background: #16213e !important;
        color: #e0e0e0 !important;
    }
    
    [role="option"]:hover {
        background: #1f77b4 !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        color: #e0e0e0 !important;
        border: 2px solid #0f3460 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        margin: 0.25rem !important;
    }
    
    /* Checkboxes */
    .stCheckbox > label {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        color: #e0e0e0 !important;
        border: 2px solid #0f3460 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* Checkbox text and labels - ensure visibility */
    .stCheckbox label span,
    .stCheckbox label p,
    .stCheckbox div[data-testid="stMarkdownContainer"] p {
        color: #e0e0e0 !important;
    }
    
    /* Checkbox input elements */
    .stCheckbox input[type="checkbox"] {
        accent-color: #1f77b4 !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
    }
    
    /* Code editor */
    .stCodeBlock {
        background: #1a1a2e !important;
        border: 2px solid #0f3460 !important;
        border-radius: 8px !important;
    }
    
    /* Ensure cursor is visible in all text areas and code editors */
    * {
        caret-color: #4da6ff !important;  /* Bright blue cursor globally */
    }
    
    /* Override for light backgrounds (if any) */
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
        caret-color: #4da6ff !important;
    }
    
    /* Input labels */
    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label,
    .stMultiSelect > label,
    .stTextArea > label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
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
    
    /* Dark theme for comparison buttons */
    .stButton button[kind="secondary"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        color: #e0e0e0 !important;
        border: 2px solid #0f3460 !important;
    }
    
    .stButton button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%) !important;
        border-color: #1f77b4 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_download_button(fig, filename, label="Download Plot (PNG, 300 DPI)", key=None):
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
        self.fd_computation_log = []  # Track what we've computed this session
        
        # Load precomputed FD coefficients (symbolic, exact)
        # This avoids numerical instability from solving Vandermonde systems
        import json
        import os
        fd_table_path = os.path.join(os.path.dirname(__file__), 'fd_coefficients.json')
        try:
            with open(fd_table_path, 'r') as f:
                self.fd_table = json.load(f)
        except:
            # Fallback: if file not found, use empty table (will compute symbolically)
            self.fd_table = {}
        
        # Load Gram polynomial data for Hermite-GP method
        # This provides much better numerical accuracy than FD
        self.gram_loaded = False
        self._GramPolyData = {}
        self._dfL_Tilde = {}
        self._dfR_Tilde = {}
        self.load_gram_data()
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
    
    def fourier_eval_with_period(self, f_hat, x, xl, period, shift=0.0):
        """Evaluate Fourier interpolant at points x with explicit period.
        
        Uses signed mode indexing for FFT coefficients.
        Accounts for grid shift parameter.
        
        Parameters:
        - f_hat: Fourier coefficients from FFT
        - x: evaluation points
        - xl: left endpoint of domain
        - period: period of the extended function
        - shift: grid shift parameter (0 <= shift <= 1)
        """
        n_coeffs = len(f_hat)
        n = n_coeffs  # Extended grid size
        h = period / n  # Grid spacing on extended domain
        
        # Account for shift: grid starts at xl + shift*h, not xl
        x_ref = xl + shift * h
        
        result = np.zeros_like(x, dtype=complex)
        
        for k in range(n_coeffs):
            # FFT indexing: k -> signed mode
            sk = k if k <= n_coeffs // 2 else k - n_coeffs
            # Phase shift accounts for grid starting at x_ref instead of xl
            result += f_hat[k] * np.exp(2j * np.pi * sk * (x - x_ref) / period)
        
        return np.real(result)
    
    def compute_extension_and_fourier(self, f_vals, xl, xr, n, c, method, r, shift=0.0, config_dict=None):
        """Compute grid extension and Fourier coefficients."""
        extended = self.extend_grid_python(f_vals, xl, xr, c, method, r, shift, config_dict)
        coeffs = np.fft.fft(extended) / len(extended)
        return extended, coeffs
    
    def extend_grid_python(self, f, xl, xr, c, method, r, shift=0.0, config_dict=None):
        """
        Python implementation of grid extension with shift parameter.
        
        Parameters:
        -----------
        config_dict : dict, optional
            Configuration dictionary containing modulation_params (for comparisons)
        """
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
        
        elif method == "Hermite" or method == "Hermite-FD":
            # Hermite extension with finite differences
            return self.extend_hermite_proper(f, c, r, shift)
        
        elif method == "Hermite-GP":
            # Hermite extension with Gram polynomials (much more accurate!)
            return self.extend_hermite_gp(f, c, r, shift)
        
        elif method == "Hermite-FD-Modulated":
            # Modulated Hermite extension with finite differences
            # Get modulation params from session state or config_dict with actual n and c
            n_for_modulation = len(f)
            mod_params = self.get_modulation_params_from_ui(r, config_dict, n_for_modulation, c) if hasattr(self, 'get_modulation_params_from_ui') else None
            return self.extend_hermite_modulated(f, c, r, shift, use_gp=False, modulation_params=mod_params)
        
        elif method == "Hermite-GP-Modulated":
            # Modulated Hermite extension with Gram polynomials
            # Get modulation params from session state or config_dict with actual n and c
            n_for_modulation = len(f)
            mod_params = self.get_modulation_params_from_ui(r, config_dict, n_for_modulation, c) if hasattr(self, 'get_modulation_params_from_ui') else None
            return self.extend_hermite_modulated(f, c, r, shift, use_gp=True, modulation_params=mod_params)
        
        else:
            raise ValueError(f"Unknown extension method: {method}")
    
    def get_modulation_params_from_ui(self, r, config_dict=None, n_actual=None, c_actual=None):
        """
        Convert UI modulation parameters to format expected by extend_hermite_modulated.
        
        Parameters:
        -----------
        r : int
            Hermite order
        config_dict : dict, optional
            Configuration dictionary (for comparisons). If None, uses st.session_state.config
        n_actual : int, optional
            Actual number of grid points (if None, uses n_min as approximation)
        c_actual : int, optional
            Actual number of extension points (if None, computed from p/q)
        
        Returns:
        --------
        dict or None : Modulation parameters with 'left' and 'right' keys
        """
        import streamlit as st
        
        # Determine which config to use
        if config_dict is not None:
            config = config_dict
        else:
            config = st.session_state.config
        
        if 'modulation_params' not in config:
            return None
        
        xl = 0.0
        xr = 1.0
        
        # Get n and c for computing extension_length
        if n_actual is not None and c_actual is not None:
            # Use provided actual values
            n = n_actual
            c = c_actual
        else:
            # Use approximation from config
            n = config.get('n_min', 16)
            p = config.get('p', 1)
            q = config.get('q', 1)
            c = int((p / q) * n)
        
        h = (xr - xl) / n
        extension_length = c * h
        
        modulation_params = {'left': [], 'right': []}
        ui_params = config['modulation_params']
        
        for m in range(r + 1):
            # Get widths from UI (as fractions 0-1)
            width_left_frac = ui_params.get(f'mod_left_{m}', (m + 1) / (r + 2))
            width_right_frac = ui_params.get(f'mod_right_{m}', (m + 1) / (r + 2))
            
            # Store as (0, width) for modulation_function
            # These represent transitions on [0,1] scaled coordinate
            modulation_params['right'].append((0.0, width_right_frac))
            modulation_params['left'].append((0.0, width_left_frac))
        
        return modulation_params
    
    def modulation_function(self, x, a, b, r=4):
        """
        Smooth modulation function using incomplete beta transition.
        
        For the interval [a, b], creates a two-stage C^(d+1) smooth transition:
        - η(x) = 1 for x < a
        - η(x) transitions through intermediate level μ at γ = (a+b)/2
        - η(x) = 0 for x > b
        
        Uses regularized incomplete beta function I_x(d+2, d+2) for smooth transitions.
        
        Parameters:
        -----------
        x : float or array
            Evaluation point(s)
        a : float
            Start of transition (η=1 for x<a)
        b : float
            End of transition (η=0 for x>b)
        r : int
            Hermite order (used to determine smoothness d = r)
        
        Returns:
        --------
        η(x) : Smooth transition from 1 to 0
        """
        x = np.atleast_1d(x)
        result = np.ones_like(x, dtype=float)
        
        if abs(b - a) < 1e-14:
            # Degenerate case: step function
            result[x >= a] = 0.0
            return result if result.shape != (1,) else float(result[0])
        
        # Parameters for the two-stage transition
        # Use d = r to ensure C^(r+1) continuity matches Hermite order
        d = max(r, 4)  # At least C^5 continuity
        mu = 0.5  # Intermediate level
        gamma = (a + b) / 2  # Join point (midpoint)
        alpha = b  # End point
        
        # First stage: [a, gamma] - transition from 1 to μ
        mask1 = (x >= a) & (x <= gamma)
        if np.any(mask1):
            x1 = (x[mask1] - a) / (gamma - a)
            # Clamp to [0, 1] to avoid numerical issues
            x1 = np.clip(x1, 0.0, 1.0)
            # I_x(d+2, d+2) = I_x(6, 6) using our implementation
            I_x1 = self._incomplete_beta_reg(x1, d + 2, d + 2)
            result[mask1] = (1 - mu) * (1 - I_x1) + mu
        
        # Second stage: [gamma, alpha] - transition from μ to 0
        mask2 = (x > gamma) & (x < alpha)
        if np.any(mask2):
            x2 = (x[mask2] - gamma) / (alpha - gamma)
            # Clamp to [0, 1]
            x2 = np.clip(x2, 0.0, 1.0)
            # I_x(d+2, d+2)
            I_x2 = self._incomplete_beta_reg(x2, d + 2, d + 2)
            result[mask2] = mu * (1 - I_x2)
        
        # Beyond alpha
        result[x >= alpha] = 0.0
        
        return result if result.shape != (1,) else float(result[0])
    
    def _incomplete_beta_reg(self, x, a, b):
        """
        Regularized incomplete beta function I_x(a, b).
        
        For symmetric case a=b, uses efficient series expansion.
        
        Parameters:
        -----------
        x : float or array
            Upper limit of integration (0 ≤ x ≤ 1)
        a, b : float
            Beta function parameters (a > 0, b > 0)
        
        Returns:
        --------
        I_x(a, b) : Regularized incomplete beta
        """
        x = np.atleast_1d(x)
        result = np.zeros_like(x, dtype=float)
        
        # For a = b (symmetric case), use series expansion
        # I_x(a, a) = 0.5 * [1 + sgn(x - 0.5) * I_{|2x-1|}(0.5, a)]
        # where I_y(0.5, a) can be computed efficiently
        
        if abs(a - b) < 1e-10:
            # Symmetric case: use direct series
            # For a = b = 6, use binomial expansion
            # I_x(6,6) = sum_{k=6}^{11} C(11,k) * x^k * (1-x)^(11-k)
            
            n = int(a + b - 1)  # 11 for a=b=6
            k_start = int(a)     # 6
            
            for k in range(k_start, n + 1):
                binom_coeff = self._binomial_coeff(n, k)
                result += binom_coeff * (x ** k) * ((1 - x) ** (n - k))
        else:
            # General case: use continued fraction or series
            # For now, use simple polynomial approximation for smoothness
            # This is less accurate but sufficient for modulation
            result = x ** a * (1 - x) ** b / self._beta_function(a, b)
        
        return result if result.shape != (1,) else float(result[0])
    
    def _binomial_coeff(self, n, k):
        """Compute binomial coefficient C(n, k)."""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        
        # Use the more efficient side
        k = min(k, n - k)
        
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        
        return result
    
    def _beta_function(self, a, b):
        """
        Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
        
        For integer or half-integer values, use factorial relations.
        """
        from math import gamma
        return gamma(a) * gamma(b) / gamma(a + b)
    
    def extend_hermite_proper(self, f, c, r, shift=0.0):
        """
        Hermite extension matching derivatives at both boundaries.
        Creates a smooth periodic continuation.
        
        Special handling for s=0 and s=1:
        - Input f has n+1 points (including both endpoints) for accurate derivatives
        - Use all n+1 points to compute derivatives
        - Drop the last point, keeping only first n points
        - Then extend those n points → total n+c points
        
        Parameters:
        - f: function values on grid (n or n+1 points)
        - c: number of extension points
        - r: Hermite order (number of derivatives to match)
        - shift: grid shift parameter (0 <= shift <= 1)
        """
        n_input = len(f)
        xl = 0.0
        xr = 1.0
        
        # Determine if we're using n+1 grid
        use_n_plus_1 = abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12
        
        if use_n_plus_1:
            # f has n+1 points, treat as n intervals
            n = n_input - 1
            h = (xr - xl) / n
            a = xl if abs(shift) < 1e-12 else xl + h
            
            # Compute derivatives using all n+1 points
            F = self.compute_fd_derivative_matrix(f, r, h, xl, xr, a)
            
            # Drop the last point (keep only first n points)
            f_trimmed = f[:n]
        else:
            # f has n points
            n = n_input
            h = (xr - xl) / n
            a = xl + shift * h
            
            # Compute derivatives using n points
            F = self.compute_fd_derivative_matrix(f, r, h, xl, xr, a)
            
            # No trimming needed
            f_trimmed = f
        
        # Extend using Hermite interpolation
        extension = np.zeros(c)
        for j in range(c):
            x = a + (n + j) * h
            extension[j] = self.hermite_eval(x, F, r, h, xr, c)
        
        # Return n points + c extension = n+c total (NOT (n+1)+c)
        return np.concatenate([f_trimmed, extension])
    
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
    
    def extend_hermite_gp(self, f, c, r, shift=0.0):
        """
        Hermite extension using Gram Polynomials for derivatives.
        
        Much more accurate than FD method (~10^-10 vs 10^-8 precision).
        
        Special handling for s=0 and s=1:
        - Input f has n+1 points (including both endpoints) for accurate derivatives
        - Use all n+1 points to compute derivatives
        - Drop the last point, keeping only first n points
        - Then extend those n points → total n+c points
        
        Parameters:
        - f: function values on grid (n or n+1 points)
        - c: number of extension points
        - r: Hermite order (r = 1 to 9)
        - shift: grid shift parameter (0 <= shift <= 1)
        """
        n_input = len(f)
        xl = 0.0
        xr = 1.0
        
        # Determine if we're using n+1 grid (MATLAB-style)
        use_n_plus_1 = abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12
        
        if use_n_plus_1:
            # f has n+1 points, treat as n intervals
            n = n_input - 1
            h = (xr - xl) / n
            a = xl if abs(shift) < 1e-12 else xl + h
            
            # Compute derivatives using all n+1 points
            F = self.compute_gp_derivative_matrix(f, r, h, xl, xr, a)
            
            # Drop the last point (keep only first n points)
            f_trimmed = f[:n]
        else:
            # f has n points
            n = n_input
            h = (xr - xl) / n
            a = xl + shift * h
            
            # Compute derivatives using n points
            F = self.compute_gp_derivative_matrix(f, r, h, xl, xr, a)
            
            # No trimming needed
            f_trimmed = f
        
        # Extend using Hermite interpolation
        extension = np.zeros(c)
        for j in range(c):
            x = a + (n + j) * h
            extension[j] = self.hermite_eval(x, F, r, h, xr, c)
        
        # Return n points + c extension = n+c total (NOT (n+1)+c)
        return np.concatenate([f_trimmed, extension])
    
    def extend_hermite_modulated(self, f, c, r, shift=0.0, use_gp=True, 
                                  modulation_params=None):
        """
        Modulated Hermite extension with spatial modulation of derivative terms.
        
        Generalizes Hermite extension by multiplying each derivative contribution
        by a smooth modulation function η(x) that controls spatial support.
        
        Standard: H(x) = Σ f^(m)(x_l)*P_l,m(x) + Σ f^(m)(x_r)*P_r,m(x)
        Modulated: H(x) = Σ f^(m)(x_l)*η_l,m(x)*P_l,m(x) + Σ f^(m)(x_r)*η_r,m(x)*P_r,m(x)
        
        Parameters:
        -----------
        f : array
            Function values on grid (n or n+1 points)
        c : int
            Number of extension points
        r : int
            Hermite order
        shift : float
            Grid shift parameter (0 <= shift <= 1)
        use_gp : bool
            If True, use Gram polynomials; if False, use finite differences
        modulation_params : dict, optional
            Parameters for modulation function. Structure:
            {
                'left': [(a1, b1), (a2, b2), ..., (a_r+1, b_r+1)],   # One (a,b) pair per derivative order m=0..r
                'right': [(a1, b1), (a2, b2), ..., (a_r+1, b_r+1)]
            }
            If None, uses default parameters based on extension size.
        
        Returns:
        --------
        array : Extended function values (n+c points)
        """
        n_input = len(f)
        xl = 0.0
        xr = 1.0
        
        # Determine if we're using n+1 grid
        use_n_plus_1 = abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12
        
        if use_n_plus_1:
            n = n_input - 1
            h = (xr - xl) / n
            a = xl if abs(shift) < 1e-12 else xl + h
            F_compute = self.compute_gp_derivative_matrix if use_gp else self.compute_fd_derivative_matrix
            F = F_compute(f, r, h, xl, xr, a)
            f_trimmed = f[:n]
        else:
            n = n_input
            h = (xr - xl) / n
            a = xl + shift * h
            F_compute = self.compute_gp_derivative_matrix if use_gp else self.compute_fd_derivative_matrix
            F = F_compute(f, r, h, xl, xr, a)
            f_trimmed = f
        
        # Set default modulation parameters if not provided
        if modulation_params is None:
            # Default: modulate each derivative order with increasing support
            modulation_params = {'left': [], 'right': []}
            
            for m in range(r + 1):
                # Higher derivatives have larger support
                # Width as fraction of [0,1] interval
                width_frac = (m + 1) / (r + 2)
                
                # Both use same width, transitioning from 0
                modulation_params['right'].append((0.0, width_frac))
                modulation_params['left'].append((0.0, width_frac))
        
        # Extend using modulated Hermite interpolation
        extension = np.zeros(c)
        per = c * h
        
        for j in range(c):
            x = a + (n + j) * h
            value = self.hermite_eval_modulated(x, F, r, h, xr, per, modulation_params)
            extension[j] = value
        
        return np.concatenate([f_trimmed, extension])
    
    def hermite_eval_modulated(self, x, F, r, h, xr, per, modulation_params):
        """
        Evaluate modulated Hermite extension at point x.
        
        Parameters:
        -----------
        x : float
            Evaluation point
        F : list
            F[0][m] = f^(m)(xr), F[1][m] = f^(m)(xl) for m = 0, ..., r
        r : int
            Hermite order
        h : float
            Grid spacing
        xr : float
            Right boundary
        per : float
            Period length (c * h)
        modulation_params : dict
            Modulation parameters with 'left' and 'right' keys
        
        Returns:
        --------
        float : Interpolated value at x
        """
        # Hermite basis evaluation (same as standard)
        x1 = x - xr
        x2 = x - xr - per
        y1 = x1 / per
        y2 = -x2 / per  # CRITICAL: y2 = -x2/per to map x2 ∈ [-per, 0] to y2 ∈ [0, 1]
        
        # Compute modulated contributions
        p1 = 0.0  # Contribution from right boundary (xr)
        p2 = 0.0  # Contribution from left boundary (xl = xr + per in periodic sense)
        
        factm = 1.0
        x1m = 1.0
        x2m = 1.0
        
        for m in range(r + 1):
            # Compute Hermite basis for this derivative order
            s1m = 0.0
            s2m = 0.0
            y1n = 1.0
            y2n = 1.0
            
            for nn in range(r - m + 1):
                c_binom = self.binomial(r + nn, r)
                s1m += c_binom * y1n
                s2m += c_binom * y2n
                y1n *= y1
                y2n *= y2
            
            # Apply modulation directly to scaled basis polynomials
            # The basis polynomials are functions of y1, y2
            # 
            # y1 ∈ [0, 1]: at xr (y1=0) → fully active, fades as y1 increases
            # y2 ∈ [-1, 0]: at xr (y2=-1) → inactive, becomes active as y2 → 0
            #
            # For y2, use abs(y2) to map [-1, 0] → [1, 0] → use as if in [0, 1]
            
            a_right, b_right = modulation_params['right'][m]
            a_left, b_left = modulation_params['left'][m]
            
            # Modulate s1m: η transitions from 1 to 0 over y1 ∈ [a_right, b_right]
            eta_1 = self.modulation_function(y1, a_right, b_right, r)
            s1m = s1m * eta_1
            
            # Modulate s2m: η transitions from 1 to 0 over y2 ∈ [a_left, b_left]
            # Now y2 ∈ [0, 1] correctly (y2 = -x2/per maps x2 ∈ [-per,0] → y2 ∈ [1,0])
            eta_2 = self.modulation_function(y2, a_left, b_left, r)
            s2m = s2m * eta_2
            
            # Add contributions
            p1 += F[0][m] * x1m * s1m / factm
            p2 += F[1][m] * x2m * s2m / factm
            
            factm *= (m + 1)
            x1m *= x1
            x2m *= x2
        
        # Blend the two polynomials
        return (y2 ** (r + 1)) * p1 + (y1 ** (r + 1)) * p2
    
    def compute_h2_sobolev_norm_periodic(self, f_extended, n, c):
        """
        Compute the H² Sobolev norm of a periodic extension using FFT.
        
        For a periodic function with Fourier coefficients f_hat[k], the H² norm is:
        ||f||_{H²}² = Σ_k (1 + |k|²)² |f_hat[k]|²
        
        This measures both the function values and their smoothness (derivatives).
        
        Parameters:
        -----------
        f_extended : array
            Extended function values (n+c points, periodic)
        n : int
            Original grid size
        c : int
            Extension size
            
        Returns:
        --------
        float : H² Sobolev norm squared
        """
        N = len(f_extended)  # n + c
        
        # Compute FFT coefficients
        f_hat = np.fft.fft(f_extended) / N
        
        # Compute H² norm: sum of (1 + k²)² |f_hat[k]|²
        # Use proper frequency indexing
        h2_norm_sq = 0.0
        for k in range(N):
            # Map to signed frequency
            freq = k if k <= N // 2 else k - N
            weight = (1 + freq**2)**2
            h2_norm_sq += weight * np.abs(f_hat[k])**2
        
        return h2_norm_sq
    
    def compute_hr_sobolev_norm_periodic(self, f_extended, n, c, r_order):
        """
        Compute the H^r Sobolev norm of a periodic extension using FFT.
        
        For a periodic function with Fourier coefficients f_hat[k], the H^r norm is:
        ||f||_{H^r}² = Σ_k (1 + |k|²)^r |f_hat[k]|²
        
        Parameters:
        -----------
        f_extended : array
            Extended function values (n+c points, periodic)
        n : int
            Original grid size
        c : int
            Extension size
        r_order : int
            Sobolev order (r=1 for H¹, r=2 for H², etc.)
            
        Returns:
        --------
        float : H^r Sobolev norm squared
        """
        N = len(f_extended)  # n + c
        
        # Compute FFT coefficients
        f_hat = np.fft.fft(f_extended) / N
        
        # Compute H^r norm: sum of (1 + k²)^r |f_hat[k]|²
        hr_norm_sq = 0.0
        for k in range(N):
            # Map to signed frequency
            freq = k if k <= N // 2 else k - N
            weight = (1 + freq**2)**r_order
            hr_norm_sq += weight * np.abs(f_hat[k])**2
        
        return hr_norm_sq
    
    def compute_max_norm_extension(self, f_extended, n, c):
        """
        Compute the max norm of the extension region.
        
        Parameters:
        -----------
        f_extended : array
            Extended function values (n+c points)
        n : int
            Original grid size
        c : int
            Extension size
            
        Returns:
        --------
        float : Max absolute value in extension region
        """
        if c <= 0:
            return 0.0
        extension = f_extended[n:]
        return np.max(np.abs(extension))
    
    def optimize_modulation_params(self, f, c, r, shift=0.0, use_gp=True, 
                                    n_grid=11, n_iterations=3, norm_type='h2', hr_order=None):
        """
        Find optimal modulation parameters by minimizing a chosen norm.
        
        Uses coordinate descent with grid search (no scipy required).
        
        Parameters:
        -----------
        f : array
            Function values on grid (n or n+1 points)
        c : int
            Number of extension points
        r : int
            Hermite order
        shift : float
            Grid shift parameter
        use_gp : bool
            Use Gram polynomials (True) or finite differences (False)
        n_grid : int
            Number of grid points for each parameter search (default: 11)
        n_iterations : int
            Number of coordinate descent iterations (default: 3)
        norm_type : str
            'h2' for H² Sobolev norm (smoothest extension)
            'hr' for H^r Sobolev norm (requires hr_order parameter)
            'max' for max norm (smallest amplitude extension)
        hr_order : int, optional
            Order for H^r norm (only used when norm_type='hr')
            
        Returns:
        --------
        dict : Optimal modulation parameters with 'left' and 'right' keys
        dict : Optimization result info
        """
        n_input = len(f)
        xl = 0.0
        xr = 1.0
        
        # Determine grid setup
        use_n_plus_1 = abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12
        if use_n_plus_1:
            n = n_input - 1
            h = (xr - xl) / n
            a = xl if abs(shift) < 1e-12 else xl + h
            f_trimmed = f[:n]
        else:
            n = n_input
            h = (xr - xl) / n
            a = xl + shift * h
            f_trimmed = f
        
        # Compute derivatives once (these don't change during optimization)
        F_compute = self.compute_gp_derivative_matrix if use_gp else self.compute_fd_derivative_matrix
        F = F_compute(f, r, h, xl, xr, a)
        
        per = c * h
        
        # Define objective function based on norm_type
        def objective(params):
            """
            Objective function for optimization.
            
            params: array of 2*(r+1) values
                    [width_right_0, ..., width_right_r, width_left_0, ..., width_left_r]
            """
            # Unpack parameters
            mod_params = {'left': [], 'right': []}
            for m in range(r + 1):
                width_right = params[m]
                width_left = params[r + 1 + m]
                mod_params['right'].append((0.0, width_right))
                mod_params['left'].append((0.0, width_left))
            
            # Compute extension
            extension = np.zeros(c)
            for j in range(c):
                x_eval = a + (n + j) * h
                extension[j] = self.hermite_eval_modulated(x_eval, F, r, h, xr, per, mod_params)
            
            # Full extended function (periodic)
            f_extended = np.concatenate([f_trimmed, extension])
            
            # Compute norm based on type
            if norm_type == 'max':
                return self.compute_max_norm_extension(f_extended, n, c)
            elif norm_type == 'hr' and hr_order is not None:
                return self.compute_hr_sobolev_norm_periodic(f_extended, n, c, hr_order)
            else:  # 'h2' or default
                return self.compute_h2_sobolev_norm_periodic(f_extended, n, c)
        
        # Initial guess: ALWAYS use the ad-hoc defaults (m+1)/(r+2) for reproducibility
        n_params = 2 * (r + 1)
        default_params = np.array([(m % (r + 1) + 1) / (r + 2) for m in range(n_params)])
        x_current = default_params.copy()
        
        # Compute initial objective (always from defaults for consistent reporting)
        initial_obj = objective(default_params)
        best_obj = initial_obj
        best_x = default_params.copy()
        
        # Grid of values to try for each parameter (finer grid for better results)
        grid_values = np.linspace(0.05, 1.0, n_grid)
        
        # Coordinate descent with grid search
        for iteration in range(n_iterations):
            improved = False
            
            for i in range(n_params):
                # Try all grid values for parameter i
                best_val = x_current[i]
                best_local_obj = best_obj
                
                for val in grid_values:
                    x_test = x_current.copy()
                    x_test[i] = val
                    obj = objective(x_test)
                    
                    if obj < best_local_obj:
                        best_local_obj = obj
                        best_val = val
                        improved = True
                
                # Update parameter with best value
                x_current[i] = best_val
                if best_local_obj < best_obj:
                    best_obj = best_local_obj
                    best_x = x_current.copy()
            
            # Early termination if no improvement
            if not improved:
                break
        
        # Also try: all parameters equal (simplified search)
        for uniform_val in grid_values:
            x_test = np.full(n_params, uniform_val)
            obj = objective(x_test)
            if obj < best_obj:
                best_obj = obj
                best_x = x_test.copy()
        
        # Extract optimal parameters
        optimal_params = {'left': [], 'right': []}
        for m in range(r + 1):
            width_right = best_x[m]
            width_left = best_x[r + 1 + m]
            optimal_params['right'].append((0.0, width_right))
            optimal_params['left'].append((0.0, width_left))
        
        # Compute improvement
        improvement = (initial_obj - best_obj) / initial_obj * 100 if initial_obj > 0 else 0
        
        # Return both the params and optimization info
        if norm_type == 'max':
            norm_name = 'Max'
        elif norm_type == 'hr' and hr_order is not None:
            norm_name = f'H^{hr_order}'
        else:
            norm_name = 'H²'
        
        opt_info = {
            'success': best_obj < initial_obj,
            'message': 'Optimization complete' if best_obj < initial_obj else 'No improvement found',
            'norm_type': norm_name,
            'norm_initial': initial_obj,
            'norm_optimal': best_obj,
            'improvement': improvement,
            'n_iterations': n_iterations
        }
        
        return optimal_params, opt_info
    
    def _get_default_modulation_params(self, r):
        """Get default (ad-hoc) modulation parameters."""
        mod_params = {'left': [], 'right': []}
        for m in range(r + 1):
            width = (m + 1) / (r + 2)
            mod_params['right'].append((0.0, width))
            mod_params['left'].append((0.0, width))
        return mod_params
    
    def compute_gp_derivative_matrix(self, f, r, h, xl, xr, a):
        """
        Compute derivatives at boundaries using Gram Polynomials.
        
        Much more accurate than finite differences, especially for high orders.
        Achieves ~10^-10 to 10^-12 precision vs ~10^-8 for FD.
        
        Returns F[0][m] = f^(m)(xr) and F[1][m] = f^(m)(xl)
        for m = 0, 1, ..., r
        """
        if not self.gram_loaded:
            # Fallback to FD if Gram data not available
            return self.compute_fd_derivative_matrix(f, r, h, xl, xr, a)
        
        n = len(f)
        d = r + 1
        
        # Check if we have enough points
        if n < d:
            # Not enough points - fallback to FD
            return self.compute_fd_derivative_matrix(f, r, h, xl, xr, a)
        
        if d not in self._GramPolyData:
            # Fallback to FD if this degree not available
            return self.compute_fd_derivative_matrix(f, r, h, xl, xr, a)
        
        F = [[0.0 for _ in range(d)] for _ in range(2)]
        
        # Compute offsets
        s = (a - xl) / h if h > 0 else 0.0
        last_point = a + (n - 1) * h
        offset_right = (xr - last_point) / h
        offset_left = -s
        
        # Scaling factors
        sh = np.array([h ** m for m in range(d)])
        
        # Right boundary - use last d points
        f_right = f[-d:]
        
        if abs(offset_right) < 1e-10:
            # Boundary at grid point
            CoeffRight = f_right @ self._GramPolyData[d][:d, :d]
            dfR = self._dfR_Tilde[d].T @ np.diag(1.0 / sh)
            derivs_right = CoeffRight @ dfR
        else:
            # Off-grid boundary - use polynomial extrapolation
            derivs_right = self.eval_gram_poly_derivs(f_right, offset_right, d, 'right', h)
        
        for m in range(d):
            F[0][m] = derivs_right[m]
        
        # Left boundary - use first d points
        f_left = f[:d]
        
        if abs(offset_left) < 1e-10:
            # Boundary at grid point
            CoeffLeft = f_left @ self._GramPolyData[d][:d, :d]
            dfL = self._dfL_Tilde[d].T @ np.diag(1.0 / sh)
            derivs_left = CoeffLeft @ dfL
        else:
            # Off-grid boundary
            derivs_left = self.eval_gram_poly_derivs(f_left, offset_left, d, 'left', h)
        
        for m in range(d):
            F[1][m] = derivs_left[m]
        
        return F
    
    def eval_gram_poly_derivs(self, f_vals, offset, d, side, h):
        """Evaluate polynomial derivatives at offset point using polynomial fit."""
        # Ensure f_vals is array and has correct length
        f_vals = np.asarray(f_vals, dtype=float)
        
        if len(f_vals) != d:
            raise ValueError(f"f_vals length {len(f_vals)} does not match d={d}")
        
        x_grid = np.arange(d, dtype=float)
        
        if side == 'right':
            eval_point = (d - 1) + offset
        else:
            eval_point = offset
        
        # Fit polynomial - use degree d-1 (requires d points)
        poly_coeffs = np.polyfit(x_grid, f_vals, d - 1)
        
        # Evaluate derivatives
        derivs = np.zeros(d)
        derivs[0] = np.polyval(poly_coeffs, eval_point)
        
        current_poly = poly_coeffs
        for m in range(1, d):
            current_poly = np.polyder(current_poly)
            if len(current_poly) > 0:
                derivs[m] = np.polyval(current_poly, eval_point) / (h ** m)
            else:
                derivs[m] = 0.0
        
        return derivs
    
    def fd_coefficients(self, m, q, a):
        """
        Compute finite difference coefficients for m-th derivative
        using q+1 points, offset by a.
        
        Uses precomputed symbolic values when available, otherwise computes
        symbolically using SymPy for exact rational arithmetic.
        Caches new coefficients for future use.
        """
        # Check if we have precomputed coefficients for integer a
        m_str, q_str = str(m), str(q)
        a_rounded = round(a, 10)  # Round to avoid floating point issues
        
        # Try to find exact match in cache (for integer and rational a)
        if m_str in self.fd_table and q_str in self.fd_table[m_str]:
            # Check for exact match in cache
            for a_key, coeffs in self.fd_table[m_str][q_str].items():
                try:
                    # Parse stored a value
                    if '/' in a_key:
                        num, den = a_key.split('/')
                        a_cached = float(num) / float(den)
                    else:
                        a_cached = float(a_key)
                    
                    if abs(a_cached - a_rounded) < 1e-12:
                        return np.array(coeffs, dtype=float)
                except:
                    continue
        
        # Not in cache - compute symbolically for exact result
        # Track if we've already computed this session to avoid spam
        comp_key = (m, q, a_rounded)
        show_message = comp_key not in self.fd_computation_log
        self.fd_computation_log.append(comp_key)
        
        # Use placeholder for temporary message that will disappear
        if show_message:
            message_placeholder = st.empty()
            with message_placeholder:
                st.info("Computing new FD coefficients...")
        
        try:
            import sympy as sp
            from fractions import Fraction
            
            # Convert a to rational for exact arithmetic
            a_frac = Fraction(a_rounded).limit_denominator(10000)
            a_sym = sp.Rational(a_frac.numerator, a_frac.denominator)
            
            # Number of points
            N = m + q
            
            # Build symbolic Vandermonde system
            # We want: sum_j c_j (a+j)^i = delta_{i,m} * m!
            # for i = 0, 1, ..., N-1
            A_sym = sp.zeros(N, N)
            b_sym = sp.zeros(N, 1)
            
            for i in range(N):
                for j in range(N):
                    A_sym[i, j] = (a_sym + j) ** i
            
            b_sym[m] = sp.factorial(m)
            
            # Solve symbolically (exact rational arithmetic)
            c_sym = A_sym.LUsolve(b_sym)
            
            # Convert to float array
            c = np.array([float(c_sym[i]) for i in range(N)], dtype=float)
            
            # Store in cache for future use
            if m_str not in self.fd_table:
                self.fd_table[m_str] = {}
            if q_str not in self.fd_table[m_str]:
                self.fd_table[m_str][q_str] = {}
            
            # Store with key as string representation of a
            if a_frac.denominator == 1:
                a_key = str(a_frac.numerator)
            else:
                a_key = f"{a_frac.numerator}/{a_frac.denominator}"
            
            self.fd_table[m_str][q_str][a_key] = c.tolist()
            
            # Save updated table to file (only once per unique coefficient set)
            if show_message:
                self.save_fd_table()
                # Show brief success message that auto-disappears
                import time
                with message_placeholder:
                    st.success("FD coefficient database updated!")
                time.sleep(2)  # Show for 2 seconds
                message_placeholder.empty()  # Clear message
            
            return c
            
        except Exception as e:
            if show_message:
                import time
                with message_placeholder:
                    st.warning(f"Using numerical method for FD coefficients")
                time.sleep(2)
                message_placeholder.empty()
            
            # Fall back to numerical computation with improved conditioning
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
    
    def save_fd_table(self):
        """Save FD coefficient table to JSON file."""
        import json
        import os
        
        fd_table_path = os.path.join(os.path.dirname(__file__), 'fd_coefficients.json')
        
        # Also try to save in user data outputs
        output_paths = [
            fd_table_path,
            '/mnt/user-data/outputs/fd_coefficients.json',
            'fd_coefficients.json'  # Current directory
        ]
        
        for path in output_paths:
            try:
                with open(path, 'w') as f:
                    json.dump(self.fd_table, f, indent=2)
                # Silent save - no UI message needed
                break
            except:
                continue
    
    def load_gram_data(self):
        """Load precomputed Gram polynomial data for Hermite-GP method."""
        if self.gram_loaded:
            return
        
        import json
        import os
        
        # Try multiple paths
        paths = [
            os.path.join(os.path.dirname(__file__), 'gram_poly_data.json'),
            os.path.join(os.path.dirname(__file__), 'gram_derivative_matrices.json'),
            '/mnt/user-data/outputs/gram_poly_data.json',
            '/mnt/user-data/outputs/gram_derivative_matrices.json',
            'gram_poly_data.json',
            'gram_derivative_matrices.json'
        ]
        
        try:
            # Load Gram polynomial projection matrices
            for path in paths:
                if 'gram_poly_data' in path:
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                            for d_str, matrix in data.items():
                                d = int(d_str)
                                self._GramPolyData[d] = np.array(matrix)
                        break
                    except:
                        continue
            
            # Load derivative matrices
            for path in paths:
                if 'gram_derivative_matrices' in path:
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                            for d_str, matrix in data['left'].items():
                                d = int(d_str)
                                self._dfL_Tilde[d] = np.array(matrix)
                            for d_str, matrix in data['right'].items():
                                d = int(d_str)
                                self._dfR_Tilde[d] = np.array(matrix)
                        break
                    except:
                        continue
            
            if self._GramPolyData and self._dfL_Tilde and self._dfR_Tilde:
                self.gram_loaded = True
        except Exception as e:
            # Silently fail - Hermite-GP will not be available
            pass
    
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
    st.markdown("## Configure Your Analysis")
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
    # HANDLE PENDING OPTIMIZATION/RESET (must happen BEFORE any widgets)
    # ==========================================================================
    if 'pending_optimization' in st.session_state:
        pending = st.session_state.pending_optimization
        if 'modulation_params' not in st.session_state.config:
            st.session_state.config['modulation_params'] = {}
        for m in range(pending['r'] + 1):
            opt_right = pending['params']['right'][m][1]
            opt_left = pending['params']['left'][m][1]
            # Update config
            st.session_state.config['modulation_params'][f"mod_right_{m}"] = opt_right
            st.session_state.config['modulation_params'][f"mod_left_{m}"] = opt_left
            # Directly set slider widget keys to new values
            st.session_state[f"slider_mod_right_{m}"] = opt_right
            st.session_state[f"slider_mod_left_{m}"] = opt_left
        st.session_state.optimal_mod_info = pending['info']
        del st.session_state.pending_optimization
    
    if 'pending_reset' in st.session_state:
        pending_r = st.session_state.pending_reset
        if 'modulation_params' not in st.session_state.config:
            st.session_state.config['modulation_params'] = {}
        for m in range(pending_r + 1):
            default_width = (m + 1) / (pending_r + 2)
            # Update config
            st.session_state.config['modulation_params'][f"mod_left_{m}"] = default_width
            st.session_state.config['modulation_params'][f"mod_right_{m}"] = default_width
            # Directly set slider widget keys to new values
            st.session_state[f"slider_mod_right_{m}"] = default_width
            st.session_state[f"slider_mod_left_{m}"] = default_width
        if 'optimal_mod_info' in st.session_state:
            del st.session_state.optimal_mod_info
        del st.session_state.pending_reset
    
    # ==========================================================================
    # SECTION 1: FUNCTION DEFINITION
    # ==========================================================================
    st.markdown("---")
    st.subheader("1. Test Function")
    
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
                with st.expander("Help: Simple Expressions"):
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
                with st.expander("Help: Python Code"):
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
                    if st.button("Save", key="save_func_btn", use_container_width=True):
                        if func_code and save_name:
                            st.session_state.saved_code_snippets[save_name] = {
                                'code': func_code,
                                'type': 'function'
                            }
                            st.success(f"Saved '{save_name}'")
                
                with col3:
                    if st.session_state.saved_code_snippets:
                        func_snippets = {k: v for k, v in st.session_state.saved_code_snippets.items() if v['type'] == 'function'}
                        if func_snippets:
                            selected = st.selectbox("Load:", list(func_snippets.keys()), key="load_func_select", label_visibility="collapsed")
                            if st.button("Load", key="load_func_btn", use_container_width=True):
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
                            st.success(f"Uploaded '{name}'!")
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
    st.subheader("2. Domain")
    
    col1, col2 = st.columns(2)
    with col1:
        xl_str = st.text_input(
            "Left boundary (xₗ)", 
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
            "Right boundary (xᵣ)", 
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
            
            # Format interval symbolically when possible
            def format_value(val):
                """Format value symbolically if it's a simple fraction or integer."""
                if val == int(val):
                    return str(int(val))
                # Check for common fractions
                for denom in [2, 3, 4, 5, 6, 8, 10]:
                    for numer in range(-10*denom, 10*denom + 1):
                        if abs(val - numer/denom) < 1e-10:
                            if numer == 0:
                                return "0"
                            elif denom == 1:
                                return str(numer)
                            elif abs(numer) == 1 and denom == 1:
                                return str(numer)
                            else:
                                return f"{numer}/{denom}"
                # Check for multiples of pi
                if abs(val / np.pi - round(val / np.pi)) < 1e-10:
                    mult = int(round(val / np.pi))
                    if mult == 0:
                        return "0"
                    elif mult == 1:
                        return "π"
                    elif mult == -1:
                        return "-π"
                    else:
                        return f"{mult}π"
                return f"{val:.2f}"
            
            xl_str = format_value(xl)
            xr_str = format_value(xr)
            ax_preview.set_title(f'Function on [{xl_str}, {xr_str}]', fontsize=11, fontweight='bold')
            
            st.pyplot(fig_preview)
            create_download_button(fig_preview, "function_preview", key="dl_func_preview")
            plt.close(fig_preview)
            
        except Exception as e:
            st.error(f"Preview error: {e}")
    
    # ==========================================================================
    # SECTION 3: EXTENSION METHOD
    # ==========================================================================
    st.markdown("---")
    st.subheader("3. Extension Method")
    
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
                ["Periodic", "Hermite-FD", "Hermite-GP", 
                 "Hermite-FD-Modulated", "Hermite-GP-Modulated"],
                index=2 if app.gram_loaded else 1,
                help="Periodic: tile the function. Hermite: smooth polynomial extension. Modulated: Hermite with optimizable spatial support."
            )
            st.session_state.config['method'] = method
            
            # Hermite order
            r = 4
            if method in ["Hermite-FD", "Hermite-GP", "Hermite-FD-Modulated", "Hermite-GP-Modulated"]:
                max_r = 9 if "GP" in method else 8
                label = "Degree (d)" if "GP" in method else "Order (r)"
                
                if "GP" in method:
                    d = st.slider(label, min_value=2, max_value=10, value=5, step=1)
                    r = d - 1  # Convert degree to order
                    
                    # Warn if d might be too large
                    n_min = st.session_state.config.get('n_min', 16)
                    if d > n_min:
                        st.warning(f"d={d} requires n ≥ {d}. Current n_min={n_min} may be too small. Increase n_min or reduce d.")
                else:
                    r = st.slider(label, min_value=2, max_value=8, value=st.session_state.config['r'], step=1)
                
                st.session_state.config['r'] = r
                
                # Show controls for modulated methods
                if "Modulated" in method:
                    st.markdown("**Modulation Parameters**")
                    st.caption("Control the spatial support of each derivative term. "
                              "Each derivative order m has a transition width that determines "
                              "where the modulation function η(x) goes from 1 to 0.")
                    
                    # Initialize modulation params in session state if not exists
                    if 'modulation_params' not in st.session_state.config:
                        st.session_state.config['modulation_params'] = {}
                    
                    # Get current p/q for extension length estimation
                    p_est = st.session_state.config.get('p', 1)
                    q_est = st.session_state.config.get('q', 1)
                    c_fraction = p_est / q_est  # c/n ratio
                    
                    # Create expander for modulation controls
                    with st.expander("Customize Modulation Widths", expanded=False):
                        st.markdown("""
                        **Transition Width:** Controls how far the modulation extends.
                        - **0.0**: Sharp cutoff at boundary
                        - **0.5**: Transitions over half the extension region
                        - **1.0**: Transitions over full extension region
                        
                        **Default:** Higher derivatives get wider support: `(m+1)/(r+2) × c`
                        """)
                        
                        modulation_params = {'left': [], 'right': []}
                        
                        # Create columns for left and right modulation
                        col_mod_left, col_mod_right = st.columns(2)
                        
                        with col_mod_left:
                            st.markdown("**Left Boundary (at x_r)**")
                            st.caption("Modulation transitions from x_r → x_r + width")
                        
                        with col_mod_right:
                            st.markdown("**Right Boundary (at x_l)**")
                            st.caption("Modulation transitions from x_l - width → x_l")
                        
                        # Create sliders for each derivative order
                        for m in range(r + 1):
                            # Default width as fraction of extension: (m+1)/(r+2)
                            default_width = (m + 1) / (r + 2)
                            
                            col_l, col_r = st.columns(2)
                            
                            with col_l:
                                # Left modulation width
                                key_left = f"mod_left_{m}"
                                
                                # Initialize if not exists
                                if key_left not in st.session_state.config['modulation_params']:
                                    st.session_state.config['modulation_params'][key_left] = default_width
                                
                                current_val_left = float(st.session_state.config['modulation_params'][key_left])
                                
                                width_left = st.slider(
                                    f"m={m} (Left)",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=current_val_left,
                                    step=0.025,
                                    key=f"slider_{key_left}",
                                    help=f"Transition width for {m}-th derivative at left boundary"
                                )
                                st.session_state.config['modulation_params'][key_left] = width_left
                            
                            with col_r:
                                # Right modulation width
                                key_right = f"mod_right_{m}"
                                
                                # Initialize if not exists
                                if key_right not in st.session_state.config['modulation_params']:
                                    st.session_state.config['modulation_params'][key_right] = default_width
                                
                                current_val_right = float(st.session_state.config['modulation_params'][key_right])
                                
                                width_right = st.slider(
                                    f"m={m} (Right)",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=current_val_right,
                                    step=0.025,
                                    key=f"slider_{key_right}",
                                    help=f"Transition width for {m}-th derivative at right boundary"
                                )
                                st.session_state.config['modulation_params'][key_right] = width_right
                        
                        # Add buttons for reset and optimize
                        st.markdown("**Optimization**")
                        col_reset, col_h2, col_hr, col_max = st.columns(4)
                        
                        with col_reset:
                            if st.button("Reset", key="reset_modulation", use_container_width=True,
                                        help="Reset to default values (m+1)/(r+2)"):
                                # Store pending reset - will be applied before sliders are created on rerun
                                st.session_state.pending_reset = r
                                st.rerun()
                        
                        # Helper function to run optimization with caching
                        def run_optimization(norm_type, hr_order=None):
                            try:
                                # Use a reasonable n for optimization (cap at 256 for speed)
                                # The optimal parameters are relatively insensitive to n
                                n_max_config = st.session_state.config.get('n_max', 64)
                                n_opt = min(n_max_config, 256)  # Cap at 256 for fast optimization
                                
                                p_opt = st.session_state.config.get('p', 1)
                                q_opt = st.session_state.config.get('q', 1)
                                c_opt = int((p_opt / q_opt) * n_opt)
                                shift_opt = st.session_state.config.get('shift', 0.0)
                                use_gp = "GP" in method
                                
                                # Create cache key based on relevant parameters
                                func_str = st.session_state.config.get('func_str', '')
                                hr_key = f"_hr{hr_order}" if hr_order else ""
                                cache_key = f"{func_str}_{r}_{p_opt}_{q_opt}_{shift_opt}_{use_gp}_{norm_type}{hr_key}_{xl}_{xr}"
                                
                                # Initialize cache if not exists
                                if 'optimization_cache' not in st.session_state:
                                    st.session_state.optimization_cache = {}
                                
                                # Check cache
                                if cache_key in st.session_state.optimization_cache:
                                    cached = st.session_state.optimization_cache[cache_key]
                                    st.session_state.pending_optimization = {
                                        'params': cached['params'],
                                        'info': cached['info'],
                                        'r': r
                                    }
                                    st.toast("Using cached optimization results")
                                    st.rerun()
                                    return
                                
                                h_opt = (xr - xl) / n_opt
                                use_n_plus_1_opt = abs(shift_opt) < 1e-12 or abs(shift_opt - 1.0) < 1e-12
                                
                                if use_n_plus_1_opt:
                                    x_opt = xl + np.arange(n_opt + 1) * h_opt
                                else:
                                    x_opt = xl + (np.arange(n_opt) + shift_opt) * h_opt
                                
                                f_opt = func(x_opt)
                                
                                norm_label = f"H^{hr_order}" if hr_order else norm_type
                                with st.spinner(f"Optimizing ({norm_label} norm) with n={n_opt}..."):
                                    optimal_params, opt_info = app.optimize_modulation_params(
                                        f_opt, c_opt, r, shift_opt, use_gp, norm_type=norm_type, hr_order=hr_order
                                    )
                                
                                # Store in cache
                                st.session_state.optimization_cache[cache_key] = {
                                    'params': optimal_params,
                                    'info': opt_info
                                }
                                
                                # Store pending optimization - will be applied before sliders are created on rerun
                                st.session_state.pending_optimization = {
                                    'params': optimal_params,
                                    'info': opt_info,
                                    'r': r
                                }
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Optimization failed: {e}")
                        
                        with col_h2:
                            if st.button("H² norm", key="optimize_h2", 
                                        use_container_width=True,
                                        help="Minimize H² Sobolev norm"):
                                run_optimization('h2')
                        
                        with col_hr:
                            # H^r where r is the Hermite order
                            if st.button(f"H^{r} norm", key="optimize_hr", 
                                        use_container_width=True,
                                        help=f"Minimize H^{r} Sobolev norm (matches Hermite order)"):
                                run_optimization('hr', hr_order=r)
                        
                        with col_max:
                            if st.button("Max norm", key="optimize_max", 
                                        use_container_width=True,
                                        help="Minimize max|extension| (smallest amplitude)"):
                                run_optimization('max')
                        
                        # Show optimization results if available
                        if 'optimal_mod_info' in st.session_state:
                            info = st.session_state.optimal_mod_info
                            if info.get('success', False):
                                norm_name = info.get('norm_type', 'H²')
                                initial = info.get('norm_initial', 0)
                                optimal = info.get('norm_optimal', 0)
                                # Show factor of improvement instead of percentage
                                factor = initial / optimal if optimal > 0 else float('inf')
                                st.success(f"{norm_name}: {initial:.2e} → {optimal:.2e} ({factor:.1f}x improvement)")
                            else:
                                st.warning(f"Optimization: {info.get('message', 'Unknown status')}")
                    
                    # Note: Modulation plots are now shown in the preview column
                    st.caption("See modulation function profiles in the preview panel")
                else:
                    # Clear modulation params if not using modulated method
                    if 'modulation_params' in st.session_state.config:
                        del st.session_state.config['modulation_params']
            
            # Show warning only if Gram data not loaded
            if "GP" in method and not app.gram_loaded:
                st.warning("Hermite-GP data files not found. Method will fall back to Hermite-FD.")
        
        else:  # Custom Code
            st.markdown("**Define custom extension:**")
            
            # Show examples
            with st.expander("Extension Examples"):
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
                if st.button("Save", key="save_ext_btn", use_container_width=True):
                    if extension_code and save_ext_name:
                        st.session_state.saved_code_snippets[save_ext_name] = {
                            'code': extension_code,
                            'type': 'extension'
                        }
                        st.success(f"Saved '{save_ext_name}'")
            
            with col3:
                if st.session_state.saved_code_snippets:
                    ext_snippets = {k: v for k, v in st.session_state.saved_code_snippets.items() if v['type'] == 'extension'}
                    if ext_snippets:
                        selected_ext = st.selectbox("Load:", list(ext_snippets.keys()), key="load_ext_select", label_visibility="collapsed")
                        if st.button("Load", key="load_ext_btn", use_container_width=True):
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
                        st.success(f"Uploaded '{ext_name}'!")
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
            # Use n_max for preview (shows the grid we ultimately care about)
            n_test = st.session_state.config.get('n_max', 64)
            if extension_mode == "No Extension":
                c_test = 0
            else:
                # Get p and q from config (they'll be set in Section 4)
                p_preview = st.session_state.config.get('p', 1)
                q_preview = st.session_state.config.get('q', 1)
                c_test = int((p_preview / q_preview) * n_test)
            
            # Get shift for grid generation
            preview_shift_val = st.session_state.config.get('shift', 0.0)
            h_test_preview = (xr - xl) / n_test
            
            # Generate grid - use n+1 points for s=0 or s=1 (MATLAB-style)
            use_n_plus_1_preview = abs(preview_shift_val) < 1e-12 or abs(preview_shift_val - 1.0) < 1e-12
            
            if use_n_plus_1_preview:
                # n+1 points including both endpoints
                x_test = xl + np.arange(n_test + 1) * h_test_preview
            else:
                # n points with shift
                x_test = xl + (np.arange(n_test) + preview_shift_val) * h_test_preview
            
            f_test = func(x_test)
            
            # Run extension
            if extension_mode == "No Extension":
                # No extension, just use the original data
                extended_test = f_test
                st.caption(f"No extension (c = 0)")
            elif extension_mode == "Built-in Methods":
                preview_shift = st.session_state.config.get('shift', 0.0)
                # Pass config explicitly for modulation parameters
                extended_test = app.extend_grid_python(f_test, xl, xr, c_test, method, r, preview_shift, st.session_state.config)
                # Clean caption showing key parameters
                if "Modulated" in method:
                    st.caption(f"n={n_test} (n_max), c={c_test}, r={r} | {method}")
                else:
                    st.caption(f"n={n_test} (n_max), c={c_test} | {method}")
            else:
                if custom_extension_func is None:
                    st.info("Define custom extension code above to see preview")
                    return None, None, None
                extended_test = custom_extension_func(f_test, c_test, xl, xr, n_test, **custom_extension_params)
                st.caption(f"Preview with n={n_test}, c={c_test} (using p={p_preview}, q={q_preview})")
            
            # Validate
            # Note: For s=0 or s=1, f_test has n+1 points
            # Hermite methods trim to n points before extending → return n+c
            # Simple methods (Zero, Constant, etc.) don't trim → return (n+1)+c
            preview_s = st.session_state.config.get('shift', 0.0)
            use_n_plus_1 = abs(preview_s) < 1e-12 or abs(preview_s - 1.0) < 1e-12
            
            # Hermite methods handle n+1 correctly, others don't
            hermite_methods = ["Hermite", "Hermite-FD", "Hermite-GP", 
                              "Hermite-FD-Modulated", "Hermite-GP-Modulated"]
            method_trims = method in hermite_methods
            
            if use_n_plus_1 and not method_trims:
                # Simple methods don't trim: expect (n+1) + c
                expected_len = (n_test + 1) + c_test
            else:
                # Hermite methods or non-n+1 grids: expect n + c
                expected_len = n_test + c_test
            
            if not isinstance(extended_test, np.ndarray):
                st.error("Must return numpy array")
            elif len(extended_test) != expected_len:
                st.error(f"Wrong length: expected {expected_len}, got {len(extended_test)}")
            elif c_test > 0 and not np.allclose(extended_test[:n_test], f_test[:n_test], rtol=1e-10):
                st.error("First n elements must equal input (first n)")
            elif not np.all(np.isfinite(extended_test)):
                st.error("Contains NaN or Inf")
            else:
                # Bilateral preview - simplified and corrected
                fig_test, ax_test = plt.subplots(figsize=(7, 4))
                h_test = (xr - xl) / n_test
                
                # Get shift from config
                preview_s = st.session_state.config.get('shift', 0.0)
                
                # Determine grid type and method behavior
                use_n_plus_1 = abs(preview_s) < 1e-12 or abs(preview_s - 1.0) < 1e-12
                hermite_methods = ["Hermite", "Hermite-FD", "Hermite-GP", 
                                  "Hermite-FD-Modulated", "Hermite-GP-Modulated"]
                method_trims = method in hermite_methods
                
                # Determine n_orig: the number of "original" points in extended_test
                if use_n_plus_1 and not method_trims:
                    n_orig = n_test + 1  # Simple methods keep all n+1 points
                else:
                    n_orig = n_test  # Hermite methods trim to n points
                
                # x_test may have n+1 points; use first n_orig for plotting
                x_grid_plot = x_test[:n_orig]
                
                if c_test == 0:
                    # No extension - just plot original grid
                    ax_test.plot(x_grid_plot, extended_test[:len(x_grid_plot)], 'bo', markersize=6,
                               label='Input grid function', zorder=5)
                    ax_test.axvline(xl, color='gray', linestyle='--', alpha=0.6, linewidth=2,
                                  label='Domain boundaries')
                    ax_test.axvline(xr, color='gray', linestyle='--', alpha=0.6, linewidth=2)
                else:
                    # With extension - extended_test has n_orig + c_test points
                    # Extended grid points (right side)
                    x_right_ext = xl + (np.arange(n_orig, n_orig + c_test) + preview_s) * h_test
                    
                    # Extended grid points (left side - bilateral)
                    x_left_ext = xl - (np.arange(c_test, 0, -1) - preview_s) * h_test
                    
                    # Full extended grid for green curve
                    x_full = np.concatenate([x_left_ext, x_grid_plot, x_right_ext])
                    f_full = np.concatenate([extended_test[n_orig:], extended_test[:n_orig], extended_test[n_orig:]])
                    
                    # Green curve for extended function
                    ax_test.plot(x_full, f_full, 'g-', linewidth=1.5, alpha=0.7, 
                               label='Extended function', zorder=3)
                    
                    # Blue circles for input grid function
                    ax_test.plot(x_grid_plot, extended_test[:n_orig], 'bo', markersize=6, 
                               label='Input grid function', zorder=5)
                    
                    # Red squares for extended grid function  
                    ax_test.plot(x_left_ext, extended_test[n_orig:n_orig+c_test], 'rs', markersize=6, 
                               label='Extended grid function', zorder=5)
                    ax_test.plot(x_right_ext, extended_test[n_orig:n_orig+c_test], 'rs', markersize=6, zorder=5)
                    
                    # Extension regions
                    ax_test.axvspan(x_left_ext[0], xl, alpha=0.2, color='yellow', label='Extension region')
                    ax_test.axvspan(xr, x_right_ext[-1], alpha=0.2, color='yellow')
                    
                    # Domain boundaries
                    ax_test.axvline(xl, color='gray', linestyle='--', alpha=0.6, linewidth=2, 
                                  label='Domain boundaries')
                    ax_test.axvline(xr, color='gray', linestyle='--', alpha=0.6, linewidth=2)
                
                ax_test.legend(fontsize=8, ncol=2, loc='best')
                ax_test.grid(True, alpha=0.3)
                
                # Update title - always shows n (even though we might use n+1 for derivatives)
                ax_test.set_title(f'Extension Preview: n={n_test}, c={c_test}', 
                                fontsize=10, fontweight='bold')
                
                ax_test.set_xlabel('x', fontsize=9)
                ax_test.set_ylabel('f(x)', fontsize=9)
                st.pyplot(fig_test)
                create_download_button(fig_test, "extension_preview", key="dl_ext_preview")
                plt.close(fig_test)
                
                # ============================================================
                # MODULATION FUNCTION PLOTS (for modulated methods)
                # ============================================================
                if "Modulated" in method and 'modulation_params' in st.session_state.config:
                    st.markdown("---")
                    st.markdown("**Modulation Function Profiles**")
                    st.caption("Shows η(y) vs y ∈ [0,1] for each derivative order")
                    
                    fig_mod, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7, 3))
                    
                    y_plot = np.linspace(0, 1, 200)
                    
                    # Plot left boundary modulations
                    ax_left.set_title("Left Boundary η(y)", fontsize=9)
                    ax_left.set_xlabel("y", fontsize=8)
                    ax_left.set_ylabel("η(y)", fontsize=8)
                    ax_left.grid(True, alpha=0.3)
                    ax_left.set_ylim([-0.05, 1.05])
                    
                    for m in range(r + 1):
                        width_left = st.session_state.config['modulation_params'].get(f"mod_left_{m}", (m+1)/(r+2))
                        # Compute modulation function
                        eta_values = np.array([app.modulation_function(y, 0.0, width_left, r) for y in y_plot])
                        ax_left.plot(y_plot, eta_values, label=f"m={m}", linewidth=1.5)
                    
                    ax_left.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
                    ax_left.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
                    ax_left.legend(loc='best', fontsize=7)
                    ax_left.tick_params(labelsize=7)
                    
                    # Plot right boundary modulations
                    ax_right.set_title("Right Boundary η(y)", fontsize=9)
                    ax_right.set_xlabel("y", fontsize=8)
                    ax_right.set_ylabel("η(y)", fontsize=8)
                    ax_right.grid(True, alpha=0.3)
                    ax_right.set_ylim([-0.05, 1.05])
                    
                    for m in range(r + 1):
                        width_right = st.session_state.config['modulation_params'].get(f"mod_right_{m}", (m+1)/(r+2))
                        # Compute modulation function
                        eta_values = np.array([app.modulation_function(y, 0.0, width_right, r) for y in y_plot])
                        ax_right.plot(y_plot, eta_values, label=f"m={m}", linewidth=1.5)
                    
                    ax_right.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
                    ax_right.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
                    ax_right.legend(loc='best', fontsize=7)
                    ax_right.tick_params(labelsize=7)
                    
                    plt.tight_layout()
                    st.pyplot(fig_mod)
                    create_download_button(fig_mod, "modulation_profiles", key="dl_mod_preview")
                    plt.close(fig_mod)
        
        except Exception as e:
            st.error(f"Preview failed: {e}")
    
    # ==========================================================================
    # SECTION 4: GRID CONFIGURATION
    # ==========================================================================
    st.markdown("---")
    st.subheader("4. Grid Configuration")
    
    # Initialize shift parameter if not exists
    if 'shift' not in st.session_state.config:
        st.session_state.config['shift'] = 0.0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
            p = st.number_input("Extension numerator (p)", min_value=1, max_value=32, 
                              value=default_p, step=1,
                              help="Extension size: c = (p/q) × n")
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
            q = st.number_input("Extension denominator (q)", min_value=1, max_value=32, 
                              value=default_q, step=1,
                              help="Extension size: c = floor((p/q) × n)")
            st.session_state.config['q'] = q
    
    with col3:
        n_min = st.number_input("n min", min_value=4, max_value=128, 
                              value=st.session_state.config['n_min'], step=4,
                              help="Minimum grid size")
        st.session_state.config['n_min'] = n_min
    
    with col4:
        n_max = st.number_input("n max", min_value=8, max_value=16384, 
                              value=st.session_state.config['n_max'], step=8,
                              help="Maximum grid size (up to 2048)")
        st.session_state.config['n_max'] = n_max
    
    with col5:
        # Initialize shift_str if not exists
        if 'shift_str' not in st.session_state.config:
            st.session_state.config['shift_str'] = "0"
        
        shift_str = st.text_input("Grid shift (s ∈ [0,1])", 
                                 value=st.session_state.config['shift_str'],
                                 help="Grid points: xⱼ = x_ℓ + (j+s)h. Examples: 0, 1/2, 1/4, 0.25")
        
        # Parse symbolic input to float
        try:
            # Try to evaluate as fraction or expression
            if '/' in shift_str:
                parts = shift_str.split('/')
                if len(parts) == 2:
                    shift = float(parts[0]) / float(parts[1])
                else:
                    shift = float(eval(shift_str))
            else:
                shift = float(shift_str)
            
            # Validate range
            if shift < 0 or shift > 1:
                st.error("Shift must be between 0 and 1")
                shift = max(0, min(1, shift))
            
            st.session_state.config['shift'] = shift
            st.session_state.config['shift_str'] = shift_str  # Store the string representation
        except Exception as e:
            st.error(f"Invalid shift value: {shift_str}. Using default 0.")
            shift = 0.0
            st.session_state.config['shift'] = 0.0
            st.session_state.config['shift_str'] = "0"
    
    if n_min >= n_max:
        st.error("n_min must be < n_max")
        return None, None, None
    
    n_levels = int(np.log2(n_max / n_min)) + 1
    
    # Display grid info with shift parameter
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"Extension: c = ({p}/{q}) × n")
    with col2:
        grid_sizes = [n_min * 2**i for i in range(n_levels)]
        st.info(f"Grids: {', '.join(map(str, grid_sizes[:5]))}{'...' if n_levels > 5 else ''}")
    with col3:
        # Check if using n+1 grid (MATLAB-style)
        use_n_plus_1 = abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12
        
        if shift == 0:
            grid_type = "Standard (n+1 pts, both endpoints)" if use_n_plus_1 else "Standard"
        elif abs(shift - 1.0) < 1e-12:
            grid_type = "Shifted (n+1 pts, both endpoints)"
        elif shift == 0.5:
            grid_type = "Open (midpoints, n pts)"
        else:
            grid_type = f"Shifted (s={shift}, n pts)"
        st.info(f"Grid: {grid_type}")
    
    # ==========================================================================
    # RUN BUTTON
    # ==========================================================================
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_analysis = st.button(
            "Run Complete Analysis",
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
                
                # Use floor formula: c = floor(p/q * n)
                c_level = int((p / q) * n_level)
                
                # Generate grid with shift parameter
                # Special case: s=0 or s=1 → use n+1 points (both endpoints) like MATLAB
                if abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12:
                    # Grid with both endpoints: n+1 points over n intervals
                    h = (xr - xl) / n_level
                    x_grid = xl + np.arange(n_level + 1) * h
                else:
                    # Standard: n points without endpoint
                    h = (xr - xl) / n_level
                    x_grid = xl + (np.arange(n_level) + shift) * h
                
                f_vals = func(x_grid)
                
                # Extend and compute Fourier
                # Pass config explicitly for modulation parameters
                extended, coeffs = app.compute_extension_and_fourier(
                    f_vals, xl, xr, n_level, c_level, method, r, shift, st.session_state.config
                )
                
                extended_period = (xr - xl) * (1 + c_level / n_level)
                
                # Evaluate on fine grid for convergence analysis
                # n_eval = 2 * n_max (evaluation grid size)
                n_fine = 2 * n_max
                x_fine = np.linspace(xl, xr, n_fine)
                f_true = func(x_fine)
                f_approx = app.fourier_eval_with_period(coeffs, x_fine, xl, extended_period, shift)
                
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
                'shift': shift,  # Add shift parameter
                'n_min': n_min,
                'n_max': n_max,
                'n_levels': n_levels
            }
            
            # Store modulation params if using modulated method
            if "Modulated" in method and 'modulation_params' in st.session_state.config:
                st.session_state.analysis_params['modulation_params'] = st.session_state.config['modulation_params'].copy()
            
            if not results_list:
                st.error("No valid grid sizes found! Adjust p and q.")
            else:
                st.success(f"Analysis complete! {len(results_list)} grids tested. Scroll down to see results.")
    
    return func, func_str, method


def compare_tab():
    """Compare tab - add methods to compare against the Setup configuration."""
    
    st.markdown("### Compare Extension Methods")
    
    # Check if analysis has been run
    if not st.session_state.results_list or not st.session_state.analysis_params:
        st.warning("Please run analysis in the **Setup & Test** tab first!")
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
    
    st.success("Using configuration from Setup & Test tab")
    
    # Display baseline configuration (read-only)
    st.markdown("#### Baseline Configuration (from Setup & Test)")
    
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
    p = params['p']
    q = params['q']
    
    if baseline_method in ['Hermite', 'Hermite-FD']:
        baseline_label = f"Hermite-FD (r={params['r']}, c=({p}/{q})n)"
    elif baseline_method == 'Hermite-GP':
        d = params['r'] + 1  # Convert r back to d for display
        baseline_label = f"Hermite-GP (d={d}, c=({p}/{q})n)"
    elif baseline_method == 'Hermite-FD-Modulated':
        baseline_label = f"Hermite-FD-Modulated (r={params['r']}, c=({p}/{q})n)"
    elif baseline_method == 'Hermite-GP-Modulated':
        d = params['r'] + 1
        baseline_label = f"Hermite-GP-Modulated (d={d}, c=({p}/{q})n)"
    elif baseline_method == 'Custom':
        baseline_label = f"Custom Extension (c=({p}/{q})n)"
    elif baseline_method == 'Zero' and p == 0:
        baseline_label = "No Extension"
    else:
        baseline_label = f"{baseline_method} (c=({p}/{q})n)"
    
    st.info(f"**Baseline Method**: {baseline_label}")
    
    # Show modulation parameters if baseline is modulated
    if 'Modulated' in baseline_method and 'modulation_params' in params:
        r_base = params.get('r', 4)
        mod_left_vals = []
        mod_right_vals = []
        for m in range(r_base + 1):
            left_val = params['modulation_params'].get(f'mod_left_{m}', (m+1)/(r_base+2))
            right_val = params['modulation_params'].get(f'mod_right_{m}', (m+1)/(r_base+2))
            mod_left_vals.append(f"{left_val:.3f}")
            mod_right_vals.append(f"{right_val:.3f}")
        st.caption(f"**Modulation widths (Left)**: [{', '.join(mod_left_vals)}]")
        st.caption(f"**Modulation widths (Right)**: [{', '.join(mod_right_vals)}]")
    
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
            st.info(f"Reset comparison for new configuration: {baseline_label}")
    
    # Add baseline to comparison results if not already there
    if baseline_label not in st.session_state.comparison_results:
        st.session_state.comparison_results[baseline_label] = {
            'config': params,
            'results': baseline_results
        }
    
    # ==========================================================================
    # METHOD SELECTION (REDESIGNED - SEPARATE CONTROLS)
    # ==========================================================================
    
    st.markdown("#### Method Selection")
    st.info(f"**Baseline**: {baseline_label} [Baseline] (always included)")
    
    st.markdown("**Configure method to add:**")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        # Method selection
        method_options = {
            "No Extension": "Zero",
            "Periodic": "Periodic",
            "Hermite-FD": "Hermite-FD",
            "Hermite-GP": "Hermite-GP",
            "Hermite-FD-Modulated": "Hermite-FD-Modulated",
            "Hermite-GP-Modulated": "Hermite-GP-Modulated"
        }
        
        selected_method_name = st.selectbox(
            "Extension Method",
            options=list(method_options.keys()),
            help="Select the extension method to compare"
        )
        
        selected_method = method_options[selected_method_name]
    
    with col2:
        # r parameter (only for Hermite methods)
        if selected_method in ["Hermite", "Hermite-FD", "Hermite-GP", "Hermite-FD-Modulated", "Hermite-GP-Modulated"]:
            max_r = 10 if "GP" in selected_method else 9
            label = "Degree (d)" if selected_method == "Hermite-GP" else "Order (r)"
            
            r_value = st.selectbox(
                label,
                options=list(range(2, max_r + 1)),
                index=2,  # Default to 4
                help="Hermite degree/order"
            )
            
            if selected_method == "Hermite-GP":
                r_value = r_value - 1  # Convert degree to order
        else:
            r_value = 0
            st.selectbox(
                "Order (r)",
                options=["N/A"],
                disabled=True,
                help="Only for Hermite methods"
            )
    
    with col3:
        # p parameter
        if selected_method_name == "No Extension":
            p_value = 0
            st.number_input("p", value=0, disabled=True, help="No extension (p=0)")
        else:
            p_value = st.number_input(
                "p",
                min_value=1,
                max_value=10,
                value=params['p'] if params['p'] > 0 else 1,
                help="Extension numerator: c = ⌊(p/q)×n⌋"
            )
    
    with col4:
        # q parameter  
        if selected_method_name == "No Extension":
            q_value = 1
            st.number_input("q", value=1, disabled=True, help="No extension (q=1)")
        else:
            q_value = st.number_input(
                "q",
                min_value=1,
                max_value=10,
                value=params['q'] if params['q'] > 0 else 1,
                help="Extension denominator: c = ⌊(p/q)×n⌋"
            )
    
    # Modulation parameters for modulated methods
    compare_mod_params = None
    if "Modulated" in selected_method:
        with st.expander("Modulation Parameters", expanded=False):
            st.caption("Set transition widths for each derivative order (0=sharp, 1=full extension)")
            
            # Initialize compare modulation params in session state if needed
            if 'compare_mod_params' not in st.session_state:
                st.session_state.compare_mod_params = {}
            
            compare_mod_params = {'left': [], 'right': []}
            
            # Determine actual r for modulated method
            actual_r = r_value
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Left**")
            with col_right:
                st.markdown("**Right**")
            
            for m in range(actual_r + 1):
                default_width = (m + 1) / (actual_r + 2)
                col_l, col_r = st.columns(2)
                
                with col_l:
                    key_l = f"compare_mod_left_{m}"
                    width_l = st.slider(
                        f"m={m}",
                        min_value=0.0,
                        max_value=1.0,
                        value=default_width,
                        step=0.025,
                        key=key_l,
                        label_visibility="collapsed" if m > 0 else "visible"
                    )
                    compare_mod_params['left'].append((0.0, width_l))
                
                with col_r:
                    key_r = f"compare_mod_right_{m}"
                    width_r = st.slider(
                        f"m={m} R",
                        min_value=0.0,
                        max_value=1.0,
                        value=default_width,
                        step=0.025,
                        key=key_r,
                        label_visibility="collapsed"
                    )
                    compare_mod_params['right'].append((0.0, width_r))
    
    # Create unique configuration key
    if selected_method_name == "No Extension":
        config_key = "No Extension"
    elif selected_method in ["Hermite", "Hermite-FD"]:
        config_key = f"Hermite-FD (r={r_value}, c=({p_value}/{q_value})n)"
    elif selected_method == "Hermite-GP":
        d_value = r_value + 1  # Convert back to degree for display
        config_key = f"Hermite-GP (d={d_value}, c=({p_value}/{q_value})n)"
    elif selected_method == "Hermite-FD-Modulated":
        # Include modulation info in key
        mod_summary = ",".join([f"{p[1]:.2f}" for p in compare_mod_params['left']]) if compare_mod_params else "default"
        config_key = f"Hermite-FD-Mod (r={r_value}, c=({p_value}/{q_value})n, w=[{mod_summary}])"
    elif selected_method == "Hermite-GP-Modulated":
        d_value = r_value + 1
        mod_summary = ",".join([f"{p[1]:.2f}" for p in compare_mod_params['left']]) if compare_mod_params else "default"
        config_key = f"Hermite-GP-Mod (d={d_value}, c=({p_value}/{q_value})n, w=[{mod_summary}])"
    else:
        config_key = f"{selected_method_name} (c=({p_value}/{q_value})n)"
    
    # Check if this configuration already exists
    config_exists = config_key in st.session_state.comparison_results
    
    col_add, col_info = st.columns([1, 2])
    
    with col_add:
        if config_exists:
            st.button(
                "Already Added",
                disabled=True,
                use_container_width=True,
                help="This exact configuration is already in the comparison"
            )
        else:
            add_config = st.button(
                "Add to Comparison",
                type="primary",
                use_container_width=True,
                help="Add this configuration to the comparison"
            )
    
    with col_info:
        if config_exists:
            st.caption("This configuration is already being compared")
        else:
            st.caption(f"Will add: **{config_key}**")
    
    # Add configuration if button clicked
    if not config_exists and 'add_config' in locals() and add_config:
        # Build configuration
        new_config = {
            'method': selected_method,
            'r': r_value,
            'p': p_value,
            'q': q_value,
            'label': config_key,
            'xl': params['xl'],
            'xr': params['xr'],
            'n_min': params['n_min'],
            'n_max': params['n_max'],
            'shift': params.get('shift', 0.0)
        }
        
        # Add modulation params if using modulated method - use compare_mod_params from UI
        if "Modulated" in selected_method and compare_mod_params is not None:
            # Convert compare_mod_params to the format expected by extend_grid_python
            mod_params_dict = {}
            for m, (_, width) in enumerate(compare_mod_params['left']):
                mod_params_dict[f'mod_left_{m}'] = width
            for m, (_, width) in enumerate(compare_mod_params['right']):
                mod_params_dict[f'mod_right_{m}'] = width
            new_config['modulation_params'] = mod_params_dict
        
        # Run computation for this configuration
        with st.spinner(f"Computing {config_key}..."):
            # Parse function from params
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
            
            app = st.session_state.app
            
            results_list = []
            n = params['n_min']
            
            while n <= params['n_max']:
                # Compute c using floor formula
                c = int((p_value / q_value) * n)
                
                # Compute grid with shift parameter
                h = (params['xr'] - params['xl']) / n
                shift = params.get('shift', 0.0)
                
                # Special case: s=0 or s=1 → use n+1 points (both endpoints)
                if abs(shift) < 1e-12 or abs(shift - 1.0) < 1e-12:
                    x_grid = params['xl'] + np.arange(n + 1) * h
                else:
                    x_grid = params['xl'] + (np.arange(n) + shift) * h
                
                f_vals = func(x_grid)
                
                # Extend and compute Fourier coefficients
                try:
                    shift_val = params.get('shift', 0.0)
                    # Use new_config for modulation_params (if present), combined with baseline params
                    config_for_extension = new_config if 'modulation_params' in new_config else params
                    extended, f_hat = app.compute_extension_and_fourier(
                        f_vals, params['xl'], params['xr'], n, c, selected_method, r_value, shift_val, config_for_extension
                    )
                except Exception as e:
                    st.error(f"Error computing {config_key}: {e}")
                    break
                
                # Evaluate on fine grid for convergence analysis
                n_fine = 2 * params['n_max']
                x_fine = np.linspace(params['xl'], params['xr'], n_fine)
                f_true = func(x_fine)
                
                period = (params['xr'] - params['xl']) * (1 + c / n)
                f_approx = app.fourier_eval_with_period(f_hat, x_fine, params['xl'], period, shift_val)
                
                # Compute errors
                abs_error = np.abs(f_approx - f_true)
                max_abs_error = np.max(abs_error)
                
                max_f_orig = np.max(np.abs(f_vals))
                max_rel_error = max_abs_error / max_f_orig if max_f_orig > 0 else 0
                
                # Extended domain error
                period_extended = (params['xr'] - params['xl']) * (1 + c / n)
                n_ext = n + c
                x_grid_ext = params['xl'] + (np.arange(n_ext) + shift) * h
                f_vals_ext = func(x_grid_ext)
                max_f_ext = np.max(np.abs(f_vals_ext))
                max_rel_error_ext = max_abs_error / max_f_ext if max_f_ext > 0 else 0
                
                results_list.append({
                    'n': n,
                    'c': c,
                    'h': h,
                    'x_grid': x_grid,
                    'f_vals': f_vals,
                    'extended': extended,
                    'coeffs': f_hat,
                    'max_abs_error': max_abs_error,
                    'max_rel_error': max_rel_error,
                    'max_rel_error_extended': max_rel_error_ext,
                    # Store detailed arrays for plotting
                    'x_fine': x_fine,
                    'f_true': f_true,
                    'f_approx': f_approx,
                    'abs_error': abs_error
                })
                
                n *= 2
            
            # Store results
            st.session_state.comparison_results[config_key] = {
                'config': new_config,
                'results': results_list
            }
            
            st.success(f"Added {config_key} to comparison!")
            st.rerun()
    
    # ==========================================================================
    # CURRENTLY COMPARING
    # ==========================================================================
    
    st.markdown("---")
    st.markdown("#### Currently Comparing")
    
    if len(st.session_state.comparison_results) == 1:
        st.info("Only baseline included. Add methods above to compare.")
    else:
        # Show current configurations with remove buttons
        configs_to_display = [(k, v) for k, v in st.session_state.comparison_results.items()]
        
        # Separate baseline and other methods
        baseline_config = None
        other_configs = []
        
        for label, data in configs_to_display:
            if label == baseline_label:
                baseline_config = (label, data)
            else:
                other_configs.append((label, data))
        
        # Row 1: Baseline only (full width or centered)
        if baseline_config:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"**[Baseline] {baseline_config[0]}**")
                # Show modulation params if baseline is modulated
                if 'modulation_params' in params and 'Modulated' in baseline_config[0]:
                    mod_vals = []
                    r_base = params.get('r', 4)
                    for m in range(r_base + 1):
                        left_val = params['modulation_params'].get(f'mod_left_{m}', (m+1)/(r_base+2))
                        mod_vals.append(f"{left_val:.2f}")
                    st.caption(f"Modulation widths: [{', '.join(mod_vals)}]")
                else:
                    st.caption("(Baseline from Setup & Test)")
        
        # Rows 2+: Other methods (3 per row) with fixed height containers
        if other_configs:
            st.markdown("**Added Methods:**")
            cols_per_row = 3
            for i in range(0, len(other_configs), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(other_configs):
                        label, data = other_configs[idx]
                        
                        with col:
                            # Use container for consistent alignment
                            with st.container():
                                # Truncate long labels for display
                                display_label = label if len(label) <= 40 else label[:37] + "..."
                                st.markdown(f"**{display_label}**")
                                if st.button(f"Remove", key=f"remove_{idx}", type="secondary", use_container_width=True):
                                    del st.session_state.comparison_results[label]
                                    st.rerun()
    
    st.markdown("---")
    
    # Clear all button
    if st.button("Clear All Comparisons", type="secondary", use_container_width=False, help="Remove all methods except baseline"):
        st.session_state.comparison_results = {
            baseline_label: {
                'config': params,
                'results': baseline_results
            }
        }
        st.rerun()
    
    # ==========================================================================
    # DISPLAY COMPARISON RESULTS
    # ==========================================================================
    
    if len(st.session_state.comparison_results) > 1:  # More than just baseline
        st.markdown("---")
        st.subheader("Comparison Results")
        
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
                'Method': label + (' [Baseline]' if is_baseline else ''),
                'Extension (c)': finest['c'],
                'Best Grid': finest['n'],
                'Max Abs Error': finest['max_abs_error'],
                'Max Rel Error': finest['max_rel_error'],
                'Avg Rate': avg_rate,
                'Quality': ('Excellent' if finest['max_rel_error'] < 1e-10 else
                           'Good' if finest['max_rel_error'] < 1e-6 else
                           'Fair' if finest['max_rel_error'] < 1e-3 else 'Poor')
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
            is_baseline_row = '[Baseline]' in str(row['Method'])
            
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
        
        st.markdown(f"**Winner**: {df.loc[winner_idx, 'Method']} | [Baseline] = Baseline from Setup")
        
        # Detailed visualizations
        st.markdown("---")
        st.markdown("#### Comparison Plots")
        
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
            
            # Create shorter label for legend (split into two lines if too long)
            short_label = label + (' (baseline)' if label == baseline_label else '')
            if len(short_label) > 30 and '(' in short_label:
                parts = short_label.split('(', 1)
                short_label = parts[0].strip() + '\n(' + parts[1]
            
            ax1.loglog(grid_sizes, rel_errors, marker=marker, linestyle='-', 
                      color=colors[idx], linewidth=linewidth, markersize=8, 
                      label=short_label, 
                      alpha=alpha)
        
        ax1.set_xlabel('Grid size (n)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Max relative error (wrt original)', fontsize=13, fontweight='bold')
        ax1.set_title('Convergence Comparison', fontsize=15, fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Right: Rate comparison
        ax2 = fig_conv.add_subplot(gs[0, 1])
        
        labels_list = []
        labels_formatted = []  # Two-line formatted labels
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
            
            # Add star to baseline
            star = ' [B]' if label == baseline_label else ''
            
            # Create short label for inside bar
            if '(' in label:
                method_part = label.split('(')[0].strip()
                short_label = f"{method_part}{star}"
            else:
                short_label = label + star
            
            labels_list.append(label)
            labels_formatted.append(short_label)
            avg_rates.append(np.mean(rates) if rates else 0)
            colors_list.append(colors[idx])
        
        bars = ax2.barh(range(len(labels_formatted)), avg_rates, color=colors_list, alpha=0.7, height=0.6)
        ax2.set_yticks(range(len(labels_formatted)))
        ax2.set_yticklabels([''] * len(labels_formatted))  # Hide y-axis labels, we'll put them inside bars
        ax2.set_xlabel('Average convergence rate', fontsize=13, fontweight='bold')
        ax2.set_title('Rate Comparison', fontsize=15, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add method labels inside the bars and rates to the right
        for i, (bar, rate, label_text) in enumerate(zip(bars, avg_rates, labels_formatted)):
            # Method label inside the bar (left-aligned)
            ax2.text(0.05, bar.get_y() + bar.get_height() / 2,
                    label_text, ha='left', va='center', fontsize=9, 
                    fontweight='bold', color='white')
            # Rate value to the right of the bar
            if rate > 0:
                ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                        f'{rate:.1f}', ha='left', va='center', fontsize=10, 
                        fontweight='bold', color=colors_list[i])
        
        # Highlight best rate
        best_rate_idx = np.argmax(avg_rates)
        bars[best_rate_idx].set_alpha(1.0)
        bars[best_rate_idx].set_edgecolor('gold')
        bars[best_rate_idx].set_linewidth(3)
        
        # Adjust x-axis to show rate labels
        max_rate = max(avg_rates) if avg_rates else 1
        ax2.set_xlim(0, max_rate * 1.15)
        
        st.pyplot(fig_conv)
        create_download_button(fig_conv, "convergence_comparison", key="dl_conv_comp")
        plt.close(fig_conv)
        
        # Plot 2: Error distribution comparison
        st.markdown("---")
        st.markdown("#### Error Distribution (Finest Grid)")
        
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
        st.markdown("#### Export Comparison Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as CSV
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download Comparison Table (CSV)",
                data=csv_data,
                file_name="fourier_comparison.csv",
                mime="text/csv"
            )
    
    else:
        st.info("Add methods above to see comparison results. The baseline from Setup & Test is already included.")
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
    # st.markdown('<h1 class="main-header">HighFIE Lab</h1>', 
    #            unsafe_allow_html=True)
    
    # st.markdown("""
    # <div class="info-box">
    # <b>High-Order Fourier Interpolation with Extension</b><br>
    #An interface for developing and testing high-order Fourier Interpolation with grid extension methods
    # </div>
    # """, unsafe_allow_html=True)
    
    # Academic/Professional Description
    # Add custom CSS for the About section
    st.markdown("""
    <style>
        /* Style the About the Method expander */
        div[data-testid="stExpander"] > details > summary {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            border: 2px solid #0f3460;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        
        div[data-testid="stExpander"] > details > summary:hover {
            background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
            border-color: #1f77b4;
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.3);
            transform: translateY(-2px);
            transition: all 0.3s ease;
        }
        
        /* Style the expanded content area */
        div[data-testid="stExpander"] > details[open] > div {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 0 0 10px 10px;
            border: 2px solid #0f3460;
            border-top: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        
        /* Style LaTeX in dark mode */
        div[data-testid="stExpander"] .katex {
            color: #e0e0e0 !important;
        }
        
        /* Style markdown text in dark mode */
        div[data-testid="stExpander"] h4,
        div[data-testid="stExpander"] p,
        div[data-testid="stExpander"] li {
            color: #e0e0e0 !important;
        }
        
        /* Style section headers */
        div[data-testid="stExpander"] h4 {
            color: #4da6ff !important;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
        
        /* Style bullet points */
        div[data-testid="stExpander"] ul {
            color: #e0e0e0 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    with st.expander("**About the Method**", expanded=False):
        st.markdown('<div style="color: #4da6ff; font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem;">Method Overview</div>', unsafe_allow_html=True)
        
        st.markdown(r"""
        Given a grid function $f$ sampled on a uniform grid of size $n$ in the interval $[x_\ell, x_r]$, 
        this tool implements Fourier interpolation with grid extension using the following approach:
        """)
        
        st.markdown("#### **Grid Structure**")
        st.markdown("The uniform grid is defined as:")
        st.latex(r"x_j = x_\ell + (j + s)h, \quad j = 0, 1, \ldots, n-1")
        st.latex(r"h = \frac{x_r - x_\ell}{n}")
        
        st.markdown(r"""
        where $s$ is a shift parameter $(0 \le s \le 1)$:
        - $s = 0$: Standard closed grid (includes $x_\ell$, uses $n+1$ points)
        - $s = \frac{1}{2}$: Open grid (grid points at interval midpoints)
        - $s \in (0,1)$: Shifted grid (customizable positioning)
        """)
        
        st.markdown("#### **Extension Process**")
        st.markdown(r"""
        1. **Data Extension**: The method first extends the grid function by adding $c$ additional points 
           to create an extended grid of size $n + c$
        2. **Extension Size**: The extension parameter $c$ is typically chosen as a rational multiple of $n$:
        """)
        st.latex(r"c = \left\lfloor \frac{p}{q} \times n \right\rfloor")
        st.markdown(r"""
           where $q$ must divide $n$ for optimal FFT efficiency
        """)
        
        st.markdown("#### **Extension Methods**")
        st.markdown(r"""
        **Periodic Extension**: Simply tiles the function periodically.
        
        **Hermite Extension** (FD or GP): Constructs a smooth polynomial extension matching 
        function values and derivatives at both boundaries up to order $r$:
        """)
        st.latex(r"E(x) = \sum_{m=0}^{r} \left[ f^{(m)}(x_r) P_m^R(x) + f^{(m)}(x_\ell) P_m^L(x) \right]")
        st.markdown(r"""
        where $P_m^R, P_m^L$ are Hermite basis polynomials. Derivatives can be computed via:
        - **FD (Finite Differences)**: One sided finite difference differentiation [1]
        - **GP (Gram Polynomials)**: Modified FC-Gram method [2]
        
        **Modulated Hermite Extension**: Enhances Hermite extension with spatially-varying 
        modulation functions $\eta_m(x)$ that control the support of each derivative term:
        """)
        st.latex(r"E(x) = \sum_{m=0}^{r} \left[ f^{(m)}(x_r) \eta_m^R(x) P_m^R(x) + f^{(m)}(x_\ell) \eta_m^L(x) P_m^L(x) \right]")
        st.markdown(r"""
        The modulation functions smoothly transition from 1 to 0, with adjustable transition widths 
        for each derivative order. The **Optimization** feature finds optimal widths by minimizing:
        - **H² norm**: Sobolev norm penalizing high frequencies
        - **Hʳ norm**: Sobolev norm of order r matching the Hermite order
        - **Max norm**: Maximum amplitude in extension region (smallest overshoot)
        """)
        
        st.markdown("#### **Fourier Interpolation**")
        st.markdown(r"""
        Based on the extended data, the method constructs a Fourier interpolant as an approximation 
        to the input grid function using the Fast Fourier Transform (FFT).
        """)
        
        st.markdown("#### **Key Parameters**")
        st.markdown(r"""
        - $n$: Original grid size
        - $c$: Extension size $= \lfloor (p/q) \times n \rfloor$
        - $p, q$: Rational parameters controlling extension ratio
        - $s$: Grid shift parameter $(0 \le s \le 1)$
        - $d$ (or $r$): Hermite degree/order — higher values match more derivatives
        - Modulation widths: Control spatial support of each derivative term (0 to 1)
        - $[x_\ell, x_r]$: Computational domain interval
        """)
        
        st.markdown("#### **References**")
        st.markdown("""
        [1] A. Anand and A. Tiwari, *Fourier interpolation with high-order polynomial extension*, 
            SIAM Journal on Scientific Computing (2024). [DOI: 10.1137/18M1232826](https://doi.org/10.1137/18M1232826)
        
        [2] A. Anand and P. Nainwal, *A modified FC-Gram approximation algorithm with provable error bounds*, 
            Journal of Scientific Computing, Volume 105, Article 8 (2025). 
            [DOI: 10.1007/s10915-025-03035-4](https://doi.org/10.1007/s10915-025-03035-4)
        """)
    
    # Main tabs - SIMPLIFIED to 2 tabs
    tab_setup, tab_compare = st.tabs([
        "Setup & Test",
        "Compare Methods"
    ])
    
    # ==========================================================================
    # TAB 1: SETUP & TEST
    # ==========================================================================
    with tab_setup:
        setup_tab(app)
        
        # Add quick test section at bottom
        st.markdown("---")
        st.markdown("### Quick Test")
        st.info("**Tip**: Test your configuration with a single method before running full comparison.")
        
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
            x_grid = selected_result['x_grid']  # Original grid used for sampling
            extended = selected_result['extended']  # Now always n+c points
            s = st.session_state.config.get('shift', 0.0)
            
            # Note: x_grid may have n+1 points (for s=0,1) but extended always has n+c
            # We use first n points of x_grid for plotting
            use_n_plus_1 = abs(s) < 1e-12 or abs(s - 1.0) < 1e-12
            x_grid_plot = x_grid[:n] if use_n_plus_1 else x_grid
            
            if c == 0:
                # No extension - just plot original grid
                ax1.plot(x_grid_plot, extended, 'bo', markersize=6,
                        label='Input grid function', zorder=5)
                ax1.axvline(xl, color='gray', linestyle='--', alpha=0.6, linewidth=2,
                           label='Domain boundaries')
                ax1.axvline(xr, color='gray', linestyle='--', alpha=0.6, linewidth=2)
            else:
                # With extension - extended has n+c points
                # Extended grid points (right side)
                x_right_ext = xl + (np.arange(n, n + c) + s) * h
                
                # Extended grid points (left side - bilateral)
                x_left_ext = xl - (np.arange(c, 0, -1) - s) * h
                
                # Full extended grid for green curve
                x_full = np.concatenate([x_left_ext, x_grid_plot, x_right_ext])
                f_full = np.concatenate([extended[n:], extended[:n], extended[n:]])
                
                # Green curve for extended function
                ax1.plot(x_full, f_full, 'g-', linewidth=1.5, alpha=0.7, 
                         label='Extended function', zorder=3)
                
                # Blue circles for input grid function
                ax1.plot(x_grid_plot, extended[:n], 'bo', markersize=6, 
                         label='Input grid function', zorder=5)
                
                # Red squares for extended grid function
                ax1.plot(x_left_ext, extended[n:n+c], 'rs', markersize=6, 
                         label='Extended grid function', zorder=5)
                ax1.plot(x_right_ext, extended[n:n+c], 'rs', markersize=6, zorder=5)
                
                # Extension regions
                ax1.axvspan(x_left_ext[0], xl, alpha=0.2, color='yellow', 
                           label='Extension region')
                ax1.axvspan(xr, x_right_ext[-1], alpha=0.2, color='yellow')
                
                # Domain boundaries
                ax1.axvline(xl, color='gray', linestyle='--', alpha=0.6, linewidth=2, 
                           label='Domain boundaries')
                ax1.axvline(xr, color='gray', linestyle='--', alpha=0.6, linewidth=2)
            
            ax1.set_xlabel('x', fontsize=11, fontweight='bold')
            ax1.set_ylabel('f(x)', fontsize=11, fontweight='bold')
            ax1.set_title(f'Extended Grid Function (n={n}, c={c})', fontsize=13, fontweight='bold')
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Function vs Approximation
            ax2 = fig1.add_subplot(gs1[0, 1])
            
            ax2.plot(selected_result['x_fine'], selected_result['f_true'], 'b-', linewidth=2, 
                    label='True function', alpha=0.7)
            ax2.plot(selected_result['x_fine'], selected_result['f_approx'], 'r--', linewidth=2, 
                    label='Fourier approximation', alpha=0.7)
            ax2.plot(x_grid, selected_result['f_vals'], 'go', markersize=5, label='Input grid function', zorder=5)
            
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
            
            # Row 3: Convergence analysis (plot)
            st.markdown("#### Convergence Analysis")
            st.caption(f"Evaluation grid: n_eval = 2 × n_max = {2 * params['n_max']} points")
            
            fig3 = plt.figure(figsize=(18, 6))
            
            # Plot 5: Convergence
            ax5 = fig3.add_subplot(1, 1, 1)
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
            
            plt.tight_layout()
            st.pyplot(fig3)
            create_download_button(fig3, "convergence_plot", key="dl_fig3_plot")
            plt.close(fig3)
            
            # Convergence table (separate figure with dynamic height)
            st.markdown("#### Convergence Results Table")
            
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
            
            # Dynamic height based on number of rows
            n_rows = len(table_data)
            table_height = max(3, 1 + n_rows * 0.5)  # At least 3 inches
            
            fig_table = plt.figure(figsize=(18, table_height))
            ax_table = fig_table.add_subplot(1, 1, 1)
            ax_table.axis('off')
            
            table = ax_table.table(cellText=table_data, colLabels=headers,
                                 cellLoc='center', loc='center',
                                 colWidths=[0.13, 0.13, 0.15, 0.15, 0.15, 0.11, 0.11])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.0)
            
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
            
            plt.tight_layout()
            st.pyplot(fig_table)
            create_download_button(fig_table, "convergence_table", key="dl_fig_table")
            plt.close(fig_table)
            
            # Download buttons for convergence analysis
            st.markdown("#### Export Convergence Data")
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
                    label="Download Table (CSV)",
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
                    label="Download Plot (PNG, 300 DPI)",
                    data=buf,
                    file_name="convergence_plot.png",
                    mime="image/png",
                    key="download_convergence_plot",
                    use_container_width=True
                )
            
            plt.close(fig3)
            
            st.success(f"Configuration tested successfully! Ready for comparison.")
            
            # ==================================================================
            # ==================================================================
            # POST-PROCESSING SECTION - DISABLED FOR V1
            # ==================================================================
            # Will be enabled in a future version
            
        else:
            st.info("Click **Run Analysis** above to test your configuration before comparing.")
    
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
    Last updated: January 24, 2026
    </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
