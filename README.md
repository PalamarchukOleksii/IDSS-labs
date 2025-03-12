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
