First give permission for executing ```train_valid_scores.sh``` script by running this command:
chmod +x train_valid_scores.sh

Then run:
./train_valid_scores.sh [your_slurm_file.out]

This script will generate two .txt files which are used by get_scores.py

The next step is to just run get_scores.py