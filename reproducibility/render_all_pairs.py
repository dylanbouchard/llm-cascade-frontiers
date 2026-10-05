"""Render all-pair benefit curves promised in the included escalation appendix."""
import os
from pathlib import Path
import figures
from manuscript_sources import DATASETS, result_path

root = Path(__file__).resolve().parent
os.chdir(root)
for dataset in DATASETS:
    figures.ESC_BEN_DIR = result_path(dataset, 'escalation_benefit')
    figures.fig_escalation_appendix(dataset)
