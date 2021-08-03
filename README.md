# Extending the Shuffle Model of Differential Privacy to Vectors: Experimental Evaluation

This repository contains all the supporting files for the **experimental section of the paper _Extending the Shuffle Model of Differential Privacy to Vectors_**,
including all the Python files necessary for anyone to repeat all of the experiments given as evidence for the results in the paper.
Note that the files **mitbih_test.csv** and **mitbih_train.csv** from the [**ECG Heartbeat Categorization Dataset**](https://www.kaggle.com/shayanfazeli/heartbeat) must be downloaded and placed in the same folder as the Python files.

## Authors

- **Mary Scott** (Mary.P.Scott@warwick.ac.uk), Department of Computer Science, University of Warwick, Coventry, UK, CV4 7AL
- **Graham Cormode** (G.Cormode@warwick.ac.uk), Department of Computer Science, University of Warwick, Coventry, UK, CV4 7AL
- **Carsten Maple** (CM@warwick.ac.uk), WMG, University of Warwick, Coventry, UK, CV4 7AL

## Instructions

- After downloading all the files as outlined above, first open and run **shuffle_heartbeat_data_collection.py**. There will be various progress bars appearing in the terminal, indicating which experiment is being run. 
- When the text "Thank you for using the Shuffle Model for Vectors" appears, the experiments have finished and all data for the final plots has been collected and saved in the same folder as the Python files. This should happen after **approximately 20-25 minutes**. Not all the files appearing in the folder at this stage may be directly useful in their own right, but some contain useful information for the plotting step below. Therefore, **all saved files should remain in the folder until the plotting step is complete**.
- At this point, open and run **shuffle_heartbeat_plot.py** to complete the final plots. These should appear within a few seconds, and should be **exactly the graphs found in the experimental section of _Extending the Shuffle Model of Differential Privacy to Vectors_**.
- The results generated from synthetic data can be generated in a similar way. To do this, complete the steps above, running the file **shuffle_synthetic_data_collection.py** in the first step, and the file **shuffle_synthetic_plot.py** in the third step.
