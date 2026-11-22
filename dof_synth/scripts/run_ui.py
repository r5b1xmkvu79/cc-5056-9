"""Launch the interactive DoF viewer on the procedural scene."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dof_synth.app import main

if __name__ == "__main__":
    main()
