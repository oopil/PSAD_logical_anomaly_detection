# Few shot Anomaly Detection using Positive Unlabeled Learning with Cycle Consistency and Co-occurrence Features
## Getting Started
* To set the python environment, use below command.
```
  $ conda env create -f environment.yml
```
* Download datasets from:  
    MVtec AD: <https://www.mvtec.com/company/research/datasets/mvtec-ad>  
    MPDD: <https://github.com/stepanje/MPDD>  
    MTD: <https://github.com/abin24/Magnetic-tile-defect-datasets.>  
    
    Notice: If you use MTD, follow below process:
    ```
    $ cd [MTD dataset path]
    $ mkdir ./MT
    $ cd MT
    $ cp utils/set_mtd.py ./
    $ python set_mtd.py
    ```
  
## Train & Evaluation
* To train and evaluate model, run run.py as follows:
* Run only evaluation (after training), run run.py with --mode test
```
  $ run.py --mode train --gpu [gpu id] --datapath [datapath] --dataset [dataset] --category [category] --few [the number of labeled samples] 
```

This code is based on anomaly detection library (anomalib): https://github.com/openvinotoolkit/anomalib
