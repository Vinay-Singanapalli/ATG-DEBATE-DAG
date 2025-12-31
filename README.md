ATG Debate DAG (CLI Debate Simulator)
====================================

Overview
--------
This project runs a debate between two agents (e.g., Scientist vs Philosopher) on a topic you enter in the terminal, for a fixed 8-round sequence, and then produces a judge summary and winner.

What you get in a run
--------------------
- Prompt: "Enter topic for debate:"
- Round-by-round output printed to the terminal
- Final judge summary + winner printed to the terminal
- A single log file written for the run (JSONL)
- A DAG diagram image generated for the run 

Requirements
------------
- Python 3.10+ recommended. 
- Ollama installed and running locally (the project uses an Ollama chat backend through LangChain). 

Setup (first time)
------------------
1) Clone the repository and enter it:

   git clone <YOUR_GITHUB_REPO_URL>
   cd <YOUR_REPO_FOLDER>

2) Create and activate a virtual environment:

   python -m venv .venv

   Windows:
   .venv\Scripts\activate

   macOS/Linux:
   source .venv/bin/activate

3) Install dependencies:

   pip install -r requirements.txt

Ollama model setup
-----------------

   ollama pull llama3.2:1b
change the version if you face any problem

How to run
----------
Start a debate:

   python run_debate.py

Then enter a topic when prompted:
   Enter topic for debate: <type your topic here>

Run with a seed (useful for testing/demo determinism):

   python run_debate.py --seed 7 

Outputs
-------
After the run finishes, the CLI prints file paths similar to:
- debate_log_<timestamp>.jsonl  (single log file for the run)
- debate_dag_<timestamp>.png    (DAG diagram image)

Tip: If you want to inspect why a turn was retried/rejected, open the JSONL log and search for rejection/coherence entries. 

Troubleshooting
---------------
- If you see model/connection errors: verify Ollama is installed, running, and the required model is pulled. 
- If the debate repeats/falls back often: check the log for rejection reasons and coherence flags to tune validation thresholds. 
