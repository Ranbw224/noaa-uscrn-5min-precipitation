"""Include the executable byte-policy fixtures in project test discovery."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from test_source_text import SourceTextTests  # noqa: F401
