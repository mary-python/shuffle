# Applying the Shuffle Model of Differential Privacy to Vector Aggregation: Experimental Evaluation

This repository contains all the supporting files for the **experimental section of the paper _Applying the Shuffle Model of Differential Privacy to Vector Aggregation_**, including all the Python files necessary for anyone to repeat all of the experiments given as evidence for the results in the paper.
Note that the files **mitbih_test.csv** and **mitbih_train.csv** from the [**ECG Heartbeat Categorization Dataset**](https://www.kaggle.com/shayanfazeli/heartbeat) must be downloaded and placed in the same folder as the Python files.

## Authors

- **Mary Scott**, Department of Computer Science, University of Warwick
- **Graham Cormode**, Department of Computer Science, University of Warwick
- **Carsten Maple**, WMG, University of Warwick

## Instructions

- After downloading all the files as outlined above, first open and run **shuffle_all_data_collection.py**, or **shuffle_all_data_collection_with_baseline.py** if interested in the baseline experiments. There will be various progress bars appearing in the terminal, indicating which experiment is being run. 
- When the text "Thank you for using the Shuffle Model for Vectors" appears, the experiments have finished and all data for the final plots has been collected and saved in the same folder as the Python files. This should happen **within 1 hour**. Not all the files appearing in the folder at this stage may be directly useful in their own right, but some contain useful information for the plotting step below. Therefore, **all saved files should remain in the folder until the plotting step is complete**.
- At this point, open and run **shuffle_all_plot.py** or **shuffle_all_plot_with_baseline.py** to complete the final plots. These should appear within a few seconds, and should be **exactly the graphs found in the experimental section of _Applying the Shuffle Model of Differential Privacy to Vector Aggregation_**.
