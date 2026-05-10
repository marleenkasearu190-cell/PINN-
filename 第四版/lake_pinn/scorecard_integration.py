# Auto-split from the run9 monolith. Keep behavior changes out of this layer.
from .common import *


def default_scorecard_script_path() -> Path:
    """Locate the shared scorecard script used by the experiment notebooks."""
    candidates = [
        Path(__file__).resolve().with_name('lake_profile_scorecard.py'),
        PROJECT_DIR.parent / '第三版' / 'lake_profile_scorecard.py',
        PROJECT_DIR / 'lake_profile_scorecard.py',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def run_scorecard_report(
    truth_csv_path,
    prediction_csv_path,
    output_dir,
    label='selected_prediction',
    report_name=None,
    scorecard_script_path=None,
):
    """Generate the layered scorecard PNG in the prediction output directory."""
    truth_csv_path = Path(truth_csv_path) if truth_csv_path is not None else None
    prediction_csv_path = Path(prediction_csv_path) if prediction_csv_path is not None else None
    output_dir = Path(output_dir)
    scorecard_script_path = Path(scorecard_script_path) if scorecard_script_path else default_scorecard_script_path()

    if truth_csv_path is None or not truth_csv_path.exists():
        return None, 'truth_missing'
    if prediction_csv_path is None or not prediction_csv_path.exists():
        return None, 'prediction_missing'
    if not scorecard_script_path.exists():
        return None, 'scorecard_script_missing'

    if report_name is None:
        report_name = f"{prediction_csv_path.stem}_scorecard_report.png"

    command = [
        sys.executable,
        str(scorecard_script_path),
        '--truth',
        str(truth_csv_path),
        '--pred',
        str(prediction_csv_path),
        '--label',
        str(label),
        '--out-dir',
        str(output_dir),
        '--report-name',
        str(report_name),
    ]

    result = subprocess.run(
        command,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f'exit_code_{result.returncode}'
        return None, message

    return output_dir / report_name, result.stdout.strip()
