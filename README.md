# Aggregation and Transformation of Vector-Valued Messages in the Shuffle Model of Differential Privacy: Experimental Evaluation

This repository contains all the supporting files for the **experimental sections** of the papers **_[Aggregation and Transformation of Vector-Valued Messages in the Shuffle Model of Differential Privacy](https://ieeexplore.ieee.org/document/9696239)_** and **_[Applying the Shuffle Model of Differential Privacy to Vector Aggregation](https://arxiv.org/abs/2112.05464)_**, including all the Python files necessary for anyone to repeat all of the experiments given as evidence for the results in the paper.

## Environment

- Install the **latest version of Python 3**, and install the additional packages **decimal, matplotlib, numpy, progress and scipy** using **PIP**.
- Download the Python files **shuffle_all_data_collection.py** and **shuffle_all_plot.py**, or **shuffle_all_data_collection_with_baseline.py** and **shuffle_all_plot_with_baseline_corrected.py** if interested in the baseline experiments.
- Download the files **mitbih_test.csv** (102.89 MB) and **mitbih_train.csv** (411.5 MB) from the [**ECG Heartbeat Categorization Dataset**](https://www.kaggle.com/shayanfazeli/heartbeat) and place them in the same folder as the Python files.

## Instructions

- After setting up the Python environment and downloading all the required files as outlined above, open and run **shuffle_all_data_collection.py** or **shuffle_all_data_collection_with_baseline.py**. There will be various progress bars appearing in the terminal, indicating which experiment is being run. 
- When the text "Thank you for using the Shuffle Model for Vectors" appears, the experiments have finished and all data for the final plots has been collected and saved in the same folder as the Python files. This should happen **within 1 hour**. Not all the files appearing in the folder at this stage may be directly useful in their own right, but some contain useful information for the plotting step below. Therefore, **all saved files should remain in the folder until the plotting step is complete**.
- At this point, open and run **shuffle_all_plot.py** or **shuffle_all_plot_with_baseline_corrected.py** to complete the final plots. These should appear within a few seconds, and should be **exactly the graphs found in the experimental sections** of **_[Aggregation and Transformation of Vector-Valued Messages in the Shuffle Model of Differential Privacy](https://ieeexplore.ieee.org/document/9696239)_** and **_[Applying the Shuffle Model of Differential Privacy to Vector Aggregation](https://arxiv.org/abs/2112.05464)_**.

## Authors

- **[Mary Scott](https://warwick.ac.uk/fac/sci/dcs/people/u1607226)**, Department of Computer Science, University of Warwick
- **[Graham Cormode](http://dimacs.rutgers.edu/~graham/)**, Department of Computer Science, University of Warwick
- **[Carsten Maple](https://warwick.ac.uk/fac/sci/wmg/people/profile/?wmgid=1102)**, WMG, University of Warwick
