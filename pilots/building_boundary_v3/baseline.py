"""Read-only imports of preserved v2/v1; no model or external assets."""
import sys
from pathlib import Path
V2=Path(__file__).resolve().parent.parent/'building_boundary_v2'
sys.path.insert(0,str(V2))
import pilot_v2 as v2
v1=v2.v1
