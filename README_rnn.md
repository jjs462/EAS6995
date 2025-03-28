**To log in to NCAR, type NCAR ARC into google- this will take you to the correct website: https://arc.ucar.edu/ You can log in at the top right**
- Log in using jessicasmith@ucar.edu
- Check your ‘Project Activity’ - your GPU usage (core hours)- this may only update once a day
- Our S2025 EAS 6995 class project code: UCOR 0090

**Log in to derecho using terminal and ssh by opening your terminal and typing:**
  
ssh -Y username@derecho.hpc.ucar.edu 

or: 

ssh username@derecho.hpc.ucar.edu ssh jessicasmith@casper.hpc.ucar.edu

- The first one with the -Y didn’t work for me: I had to remove it
- username is jessicasmith
- Sometimes it didn’t accept my password even though it was correct
- Replace ‘derecho’ with ‘casper’ in the terminal- removing the -Y worked for me here as well
- To test the terminals once run, type echo hello and the terminal prints hello
- echo $SHELL tells you what shell you’re in
- Make sure you're in the correct directory: **/glade/work/jessicasmith/eas6995**
- To make a file (.pbs) and submit it as a job then check on it (using casper):
    - vim simple_job.pbs: type i to get into edit mode
        - #!/bin/bash
        - #PBS -N SimpleJob
        - #PBS -l select=1:ncpus=1:mem=4GB
        - #PBS -l walltime=00:30:00
        - #PBS -j oe
        - #PBS -A UCOR0090
        - #PBS -q casper
        - module load conda
        - python my_awesome_script.py
    - Esc :wq (saves and quits)
    - Run using this command: qsub simple_job.pbs
    - Check on the job with this: qstat -u jessicasmith

**For the jupyter hub: log in using jessicasmith and duo push** https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/jupyterhub/
- You can check if the projects you are running via jupyter hub are running via the terminal- that is why there may be a project running in the terminal when the one you submitted in the terminal is finished
  - Click ‘start’ under the ‘actions’ tab
  - You can upload current files from your computer by clicking the small upward arrow button
  - Pytorch was not loading in the original kernel (or conda), but when entering the commands below into a new Jupyter Terminal then selecting the new kernel, it worked. 

module load conda

conda create -n torch_env python=3.10 pytorch torchvision torchaudio cpuonly -c pytorch -y

conda install ipykernel -y

- To upload files: scp local_file username@derecho.hpc.ucar.edu:/path/to/hpc/folder
- To download files: scp username@derecho.hpc.ucar.edu:/path/to/hpc/folder local_file
- In the Jupyter terminal:
  
  module load conda
  
  conda create -n my-npl-vo -- clone npl
  
  - Replace my-npl-vo with eas6995 and - - with two dashes next to each other (Google Chrome turns them into a single big dash)
  - Next time you log in use: conda activate eas6995-ncar 
  - You can use Jupyter terminal as the code cell and the computer terminal as the place where you run the file and where it produces outputs. Run a .sh (or .pbs file with the command sh NameOfFile

**You can have a similar set up on VSCode:**
- Connect Visual Studio Code to NCAR HPC systems using the Remote SSH extension: https://code.visualstudio.com/docs/remote/ssh 
**Monitor jobs:**
- Check current jobs type this into the terminal: qstat -u jessicasmith
- View core hours: ncar_accounting_report

**To clone your git repository: git clone https://github.com/your_username/your_repo.git**

**Notes of NumPy implementation of the RNN exercise:**
- Understanding the Model
  - Wxh: Weight matrix from input to hidden layer.
  - Whh: Weight matrix from previous hidden state to current hidden state (recurrent connection).
  - Why: Weight matrix from hidden layer to output.
- Tanh activation function:
  - Maps values between -1 and 1, centered around 0.
  - Preferred over sigmoid because it's zero-centered and gradients can move in both directions.
  - Helps reduce vanishing gradient issues better than sigmoid.
- Hidden state initialization:
  - Initialized with zeros (zero set placeholders) matching the shape of the hidden state.
  - Acts as the starting point at time step t-1 for training.
- Training Process
- Loss function:
  - Uses cross-entropy loss: -np.log(ps[t][targets[t], 0]).
  - Effective for classification tasks like predicting the next character.
- Weight updates:
  - Model processes input one character at a time.
  - Tries to predict the next character based on previous input.
  - Learns to generate outputs with real words in Shakespearean style, even if nonsensical.
- Gradient clipping (np.clip):
  - Prevents exploding gradients by capping gradient values.
  - Stabilizes training.
