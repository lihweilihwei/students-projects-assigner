# Students Projects Assigner

This project automates the assignment of students to project topics based on their ranked preferences. It uses a lottery and optimization algorithm to ensure fair and efficient assignments, even when there are more topics than students or overlapping preferences.

## Features
- Handles incomplete, duplicated, and empty preferences
- Supports more topics than students
- Uses a lottery for first-choice conflicts
- Assigns unranked topics if necessary
- Outputs assignment and summary CSV files

## Setup Instructions

### 1. Install Conda (if not already installed)
Download and install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### 2. Create the Conda Environment
Open Anaconda Prompt and run:
```
conda env create -f environment.yml -n students-projects-assigner
conda activate students-projects-assigner
```

### 3. Prepare Student Preferences
- Edit `prefs.csv` to include student names and their ranked topic preferences.
- **Privacy Warning:** Do NOT commit real student names or sensitive data to public repositories. Use pseudonyms or anonymized IDs.
- Add `prefs.csv` to `.gitignore` to prevent accidental uploads:
	```
	prefs.csv
	```

### 4. Update Assigner Configurations
- Edit `assigner.py` to configure key variables such as:
    - `PREFS_CSV`: Path to the student preferences CSV file (default: `'prefs_sample.csv'`)
    - `SAVE_PREFIX`: Name of the assignment output file (default: `'assignment.csv'`). Also outputs a summary file (default: `'assignment_summary.csv'`)
    - `SEED`: Integer seed for reproducible results (default: `None`)
    - `RANK_TO_COST`: Preference ranking system translating ranks into costs to minimize; lower numbers indicate higher preference.
- Update `RANK_TO_COST` to change the number of rank levels. `PREFS_CSV` must have column names `'rank1', 'rank2', ...` 

### 5. Run the Assigner
In the activated environment, run:
```
python assigner.py
```

### 6. Review Results
- `assignment.csv`: Contains the assignment of students to topics, with columns for lottery winners and unranked assignments.
- `assignment_summary.csv`: Summary statistics of the assignment process.

## Protecting Student Privacy
- Always anonymize student data in `prefs.csv` before sharing or uploading.
- Keep `prefs.csv` and other data files out of version control by listing it in `.gitignore`. All files are already ignored except where specified in `.gitignore`. 
- Delete or encrypt files containing sensitive information when no longer needed.

## Troubleshooting
- If `conda` is not recognized, ensure Anaconda/Miniconda is installed and added to your PATH.
- For help, contact the project maintainer or your course instructor.

---

**This tool is for educational use. Protect student privacy at all times.**

---

*Portions of this project and documentation were assisted by ChatGPT.*