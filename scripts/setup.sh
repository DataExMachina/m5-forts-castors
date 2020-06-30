echo "Create .venv and install package"
python3.6 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
deactivate

echo "Setup done"
