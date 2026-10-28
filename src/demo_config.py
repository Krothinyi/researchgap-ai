
import os
from pathlib import Path


# ==========================================
# PUBLIC PORTFOLIO DEMO CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DEMO_MODE = os.getenv(
    "RESEARCHGAP_DEMO_MODE", "false"
).strip().lower() in {"true", "1", "yes", "on"}


# ==========================================
# DATA STORAGE CONFIGURATION
# ==========================================

if DEMO_MODE:
    DATA_DIR = BASE_DIR / "demo_runtime"
else:
    DATA_DIR = BASE_DIR / "data"

QDRANT_PATH = DATA_DIR / "qdrant_storage"

MANIFEST_PATH = DATA_DIR / "downloaded_papers.json"


# ==========================================
# WRITE PROTECTION
# ==========================================

def require_write_permission(operation="This operation"):
    """
    Block changes to shared research data
    when public demo mode is enabled.
    """

    if DEMO_MODE:
        raise PermissionError(
            f"{operation} is disabled in the "
            "public portfolio demo."
        )
