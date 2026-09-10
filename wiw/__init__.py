"""World in World: training-free re-cinematography, bullet time and editing of real videos
with a frozen causal video world model."""
import os
import sys

_LINGBOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lingbot_world")
if _LINGBOT not in sys.path:
    sys.path.insert(0, _LINGBOT)

__version__ = "0.1.0"
