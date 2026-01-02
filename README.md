# Toxic Conversation Detection with BERT

This repository contains a project for detecting toxic language in Italian conversations using BERT.
The approach includes both full-conversation classification and single-message analysis with conversational context,
with the goal of identifying the most toxic messages within each dialogue using the same initially trained model.

## Reproducibility Instructions

**1. Clone the repository on your local machine.**

   
    git clone https://github.com/alessandroliuzzi/CIPV-2024-25.git
    cd CIPV-2024-25
  

**2. Create a Python virtual environment.**

    
    python -m venv venv
    

**3. Activate the virtual environment.**

Windows (PowerShell):

   
    .\venv\Scripts\Activate.ps1
    

Windows (cmd):
   
    venv\Scripts\activate.bat
  

macOS / Linux:
   
    source venv/bin/activate
   

**4. Install all required dependencies.**

  
    pip install -r requirements.txt
   

**5. Download and locally cache the pretrained BERT model.**

    
    python model_setup.py
   

**6. Run the main experiment.**

   
    python main.py
   

The script performs:
- initial fine-tuning of a BERT-based classifier on the dataset
- testing on full conversations,
- evaluation using standard classification metrics,
- single-message toxicity analysis with conversational context (same model),
- evaluation with the same metrics, plus identification of the most toxic message in each conversation,
- qualitative inspection through example outputs.


### GPU Support (Optional)

The project can run on a CUDA-enabled NVIDIA GPU for faster training/inference. Install PyTorch with CUDA following the official guide: https://pytorch.org/get-started/locally/.
If no GPU is available, the code runs on the CPU without issues. No separate CUDA installation is needed beyond the PyTorch libraries, but make sure your Nvidia drivers are compatible and up-to-date.


