# Instructions to Run

### 1. Clone the repository
Clone the repository to your local machine and navigate into the project directory:
```bash
git clone https://github.com/SaiTeja6304/multilingual-smile-metric-qna-eval.git
cd multilingual-smile-metric-qna-eval
```

### 2. Set up a virtual environment
Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
Install the required Python packages from the requirements file:
```bash
pip install -r requirements.txt
```

pip uninstall torch torchvision torchaudio tensorflow -y

pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

pip install tensorflow==2.21.0



### 4. Configure paths
Before running the SLURM job, you need to update the file paths in the `smile_for_llm.sh` script to match your environment. Open `smile_for_llm.sh` and modify the following directory variables:
- `REPO`: Path to this cloned repository.
- `GT_DIR`: Path to the directory containing ground truth JSONL files.
- `PRED_DIR`: Path to the directory containing prediction JSONL files.
- `OUT_DIR`: Path where the evaluation CSV results should be saved.

### 5. Submit the SLURM job
Once the paths are configured, submit the batch job to the SLURM cluster:
```bash
sbatch smile_for_llm.sh
```