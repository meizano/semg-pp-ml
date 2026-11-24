"""
Google Colab Execution Helper Script.
Provides setup instructions and execution flow for running sEMG pipeline on Google Colab GPU.
"""
from __future__ import annotations



def get_colab_setup_instructions() -> str:
    return """
# =====================================================================
# Google Colab Setup Instructions for sEMG Platform
# =====================================================================
# Copy and run the following cells in Google Colab:

# 1. Clone repository & install dependencies
!git clone https://github.com/meizano/semg-pp-ml.git
%cd semg-pp-ml
!pip install -e ".[dl]"

# 2. Mount Google Drive (for dataset and model saving)
from google.colab import drive
drive.mount('/content/drive')

# 3. Link the dataset from Google Drive (default location is data/ninapro_db8)
# !ln -s /content/drive/MyDrive/NinaPro_DB8 ./data/ninapro_db8

# 4. Train with the semg command (uses the GPU when Colab provides one)
!semg train --prep wavelet --model cnn1d --epochs 30
"""


if __name__ == "__main__":
    print(get_colab_setup_instructions())
