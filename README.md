# Figure Generation Scripts

This repository contains the Python script used to generate the manuscript figures from the current analytical workflow.

## Files

- `run.py`: main script for figure generation
- `requirements.txt`: Python package requirements

## Environment

Tested with Python 3.7.16.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run

Execute:

```bash
python run.py
```

## Output

Running `run.py` writes the following figure files to the working directory:

- `Fig5J_HBV_Macaron.png`
- `Fig5K_COVID_Macaron.png`
- `Fig5M_BIC_Macaron.png`
- `Fig5O_Stacked_Macaron.png`
- `Suppl_COVID_SampleSize_ModelEvolution.png`
- `Suppl_Sy_HBV_COVID_LearningCurve_Stability.png`

## Notes

- The current script uses data embedded in `run.py`; no external input data files are required for execution.
- Figures are saved at high resolution (`dpi=600`).
- The plotting style requests the `Arial` font. If `Arial` is unavailable on another machine, Matplotlib may substitute a different font.
