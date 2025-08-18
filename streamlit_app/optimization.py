"""
Optimization-related functions for the Streamlit application.
"""

import logging
import optuna
import optuna.visualization

logger = logging.getLogger(__name__)


def display_best_parameters(study_results):
    """Display best parameters from optimization study - handles both individual and complete backtest formats."""
    if not study_results:
        return {"error": "No optimization study available"}

    # Check if we have direct best_params (complete backtest format)
    if "best_params" in study_results and study_results["best_params"]:
        best_params = study_results["best_params"]
        best_value = study_results.get("best_value", None)
        # Try to get study if available for parameter importance
        study = study_results.get("study", None)
    # Else handle individual optimization format
    elif "study" in study_results:
        study = study_results["study"]
        if not hasattr(study, "best_params") or not study.best_params:
            return {"error": "No best parameters found"}
        best_params = study.best_params
        best_value = getattr(study, "best_value", None)
    else:
        return {"error": "No optimization study available"}

    # Get parameter importance if available
    param_importance = {}
    if study:
        try:
            param_importance = optuna.importance.get_param_importances(study)
        except Exception:
            param_importance = {}

    # Get trial statistics if study available
    total_trials = 0
    completed_trials = 0
    if study:
        completed_trials = len(
            [
                trial
                for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
            ]
        )
        total_trials = len(study.trials)

    return {
        "best_parameters": best_params,
        "best_objective_value": best_value,
        "parameter_importance": param_importance,
        "total_trials": total_trials,
        "completed_trials": completed_trials,
        "success_rate": (completed_trials / total_trials * 100) if total_trials else 0,
    }
