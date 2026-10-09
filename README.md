# EuroSAT Vision Project

A satellite image classification project using the EuroSAT RGB dataset and a ResNet-18 model. It includes baseline training and scripts for reviewing model errors, especially confusion between farmland classes.

## Project Structure

```text
EuroSAT Vision Project/
├── .vscode/                  # VS Code settings
├── data/                     # Dataset files
├── results/
│   └── baseline/             # Baseline results
├── src/
│   ├── analyze_errors.py     # Analyze prediction errors
│   ├── common.py             # Shared functions and settings
│   ├── download.py            # Download the dataset
│   ├── inspect_images.py      # View and inspect images
│   ├── make_control_sheet.py  # Prepare images for manual review
│   ├── review_annualcrop.py   # Review AnnualCrop predictions
│   ├── summarize_control.py   # Summarize the control review
│   ├── summarize_review.py    # Summarize the image review
│   ├── train.py               # Train the model
│   └── trying_extra.py        # Extra experiments
├── Assignment_Research       # Part B report
├── .gitignore                # Files Git should ignore
├── best_model.pth            # Saved model file
└── README.md                 # Project guide
```

## Project Goals

- Classify EuroSAT RGB satellite images into 10 land-use classes.
- Measure model performance on separate training, validation, and test sets.
- Review model errors and explore why farmland classes may be confused.
- Compare unclear-label rates in incorrect and correct predictions.

## Baseline Results

The baseline used ResNet-18, seed 42, and a 70/15/15 train-validation-test split.

| Metric | Result |
|---|---:|
| Validation accuracy | 96.44% |
| Test accuracy | 97.06% |
| Training images | 18,900 |
| Validation images | 4,050 |
| Test images | 4,050 |

## Error Review

AnnualCrop was often confused with Pasture and PermanentCrop. A manual review tested whether unclear or possibly incorrect labels could explain some errors.

In the control review:

- **Incorrect predictions:** 14 of 30 images (46.7%) had unclear labels.
- **Correct predictions:** 13 of 30 images (43.3%) had unclear labels.

The difference was only 3.4 percentage points. This small review did not show that unclear labels were the main cause of the model's errors.

## Setup

1. Open a terminal in the project folder.
2. Set up a Python environment with PyTorch and the other packages required by the scripts.
3. Place or download the EuroSAT RGB dataset in the expected `data/` location.
4. Check the source files for required packages and file paths. A `requirements.txt` file is not shown in the project structure.

## Running the Scripts

Run commands from the project root:

```bash
python src/download.py
python src/inspect_images.py
python src/train.py
python src/analyze_errors.py
python src/review_annualcrop.py
python src/make_control_sheet.py
python src/summarize_review.py
python src/summarize_control.py
```

Some scripts may depend on files created by earlier steps. Check each script's input paths and settings before running it. The exact purpose of `trying_extra.py` should be confirmed from its contents.

## Saved Model

`best_model.pth` is the saved model file. Load it using the model architecture and checkpoint format used in `train.py`.

## Limitations and Next Steps

- The manual review covered a small sample, so it may not represent the full dataset.
- Manual labels can be subjective.
- The planned near-duplicate image test was not completed, so no conclusion about data leakage can be made.
- A useful next step is to compare similar-image removal with random-image removal and repeat training with different seeds.
