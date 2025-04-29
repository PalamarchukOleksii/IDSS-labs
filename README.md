# IDSS Labs Project

This repository contains several lab projects, each with its own Python scripts and dependencies, focusing on Intelligent Decision Support Systems (IDSS).

## Getting Started

Follow the instructions below to set up the project.

### 1. Clone the Repository

```bash
git clone https://github.com/PalamarchukOleksii/IDSS-labs.git
```

### 2. Navigate to the Project Directory

```bash
cd IDSS-labs
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

- On **Windows**:
  ```bash
  .\.venv\Scripts\activate
  ```
- On **macOS/Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 5. Install Dependencies

Each lab has its own `requirements.txt` file. To install dependencies for a specific lab, navigate to that lab's folder and install its requirements.

For example, for `lab1`:

```bash
cd lab1
pip install -r requirements.txt
```

Repeat this for each lab as needed.

### 6. Running the Scripts

Navigate to the folder of the lab you wish to run and execute the desired Python script.

For example, for `lab1`:

```bash
python lab1.py
```

Repeat this for other labs as needed.

### 7. Deactivate the Virtual Environment

Once you're done, deactivate the virtual environment:

```bash
deactivate
```

---

## Special Instructions for Lab 4 and Lab 5

### Kaggle API Key Required

**Both Lab 4 and Lab 5 depend on datasets from Kaggle. You must set up your Kaggle API key before running any scripts in these labs.**

To do this:

1. Go to your Kaggle account settings: https://www.kaggle.com/account
2. Scroll down to the API section and click **"Create New API Token"**.
3. A `kaggle.json` file will be downloaded.
4. Place this file in the appropriate location:
   - **Linux/macOS**: `~/.kaggle/kaggle.json`
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
5. On **Linux**, set the correct file permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

Without this setup, Lab 4 and Lab 5 will not be able to access the necessary datasets.

### Viewing TensorBoard for Lab 4 and Lab 5

If you want to monitor training progress with TensorBoard in both Lab 4 and Lab 5:

1. Run the respective lab script (e.g., `lab4.py` or `lab5.py`):
   ```bash
   python lab4.py  # For Lab 4
   ```
2. Then, start TensorBoard from the lab directory:
   ```bash
   tensorboard --logdir=logs
   ```
3. Open your browser and go to `http://localhost:6006` to view the dashboard.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
