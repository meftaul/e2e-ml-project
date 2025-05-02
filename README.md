# Environment Setup
```bash
mkdir e2e-ml-project && cd e2e-ml-project
code .
```

**VSCode Extensions**
- Python
  - Python debugger
  - Python environment manager
- Jupyter
- rainbow brackets
- csv edit
- excel viewer
- Marp for VS Code

**Check python version**
```bash
python --version
```

**Python Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate
python --version
pip --version
```

**Install Jupyter Notebook and run it**
```bash
pip install jupyter
# jupyter notebook
```

**`git` Setup**
```bash
git init
# git config --global user.name "Your Name"
# git config --global user.email "your@email"
``` 

**Create a `.gitignore` file**
```bash
touch .gitignore
```

**First Commit**
```bash
git add .
git commit -m "Initial commit"
```

**Install the packages**
```bash
pip install -r requirements.txt
```