from pathlib import Path

def load_paths():

    BASE_DIR = Path(__file__).resolve().parent.parent.parent  # adjust if needed

    paths = {
        "beta_dir": BASE_DIR / "data" / "beta_dir" / "study_1",
        "results_dir": BASE_DIR / "results" / "study_1",
        "receptor_dir": BASE_DIR / "data" / "receptors",
    }
    return {k: str(v) for k, v in paths.items()}



