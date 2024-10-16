import joblib
from sklearn.pipeline import Pipeline
from lspb_model.config.core import TRAINED_MODEL_DIR
import logging


_logger = logging.getLogger(__name__)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""
    print(TRAINED_MODEL_DIR)
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model
