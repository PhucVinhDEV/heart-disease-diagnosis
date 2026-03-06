"""
Heart Disease Diagnosis - Source Code Package

This package contains core modules for the heart disease prediction system.
"""

__version__ = "1.0.0"

# Import and expose model_functions classes for easier access
try:
    from .model_functions import BasicFE, EnhancedFE, PolyFE
    __all__ = ['BasicFE', 'EnhancedFE', 'PolyFE']
except ImportError:
    # If import fails, just pass - this prevents errors during build
    pass
