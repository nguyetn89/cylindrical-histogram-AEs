# cylindrical-histogram-AEs
An implementation of the paper "Estimating Gait Normality Index based on Point Clouds using Deep Neural Network"

## Requirements
* Python
* Numpy
* TensorFlow
* Scikit-learn

## Notice
* The code was implemented to directly work on [DIRO gait dataset](http://www-labs.iro.umontreal.ca/~labimage/GaitDataset/)
* Please download the [histogram data](http://www.iro.umontreal.ca/~labimage/GaitDataset) and put the npz file into the folder **dataset**

## Usage
Process default training and test sets
```
python3 main.py
```
Specify test subject for leave-one-out cross-validation and save results
```
python3 main.py -l 0 -f results.csv
```
* -l: index of test subject (0 to 8 for 9 subjects in DIRO gait dataset)
* -f: file for saving results

Test subject | Segment length | AUCs of (6 models and the average score)
:---: | ---: | :---:
 0 |    1 | AUC values
 0 |  120 | AUC values
 0 | 1200 | AUC values

## Example of output
Default training and test sets
```
training subjects: [0 2 4 5 8]
data shape:
(6000, 256)
(4800, 256)
(38400, 256)

epoch 100: sigmoid = (0.016, 0.016), tanh = (0.008, 0.011), lrelu = (0.008, 0.012)
epoch 200: sigmoid = (0.016, 0.016), tanh = (0.008, 0.012), lrelu = (0.008, 0.012)
epoch 300: sigmoid = (0.016, 0.016), tanh = (0.008, 0.012), lrelu = (0.008, 0.012)
epoch 400: sigmoid = (0.016, 0.016), tanh = (0.008, 0.011), lrelu = (0.007, 0.012)
epoch 500: sigmoid = (0.016, 0.016), tanh = (0.008, 0.011), lrelu = (0.007, 0.012)
epoch 600: sigmoid = (0.016, 0.016), tanh = (0.008, 0.012), lrelu = (0.007, 0.012)
epoch 700: sigmoid = (0.016, 0.016), tanh = (0.008, 0.011), lrelu = (0.007, 0.012)
epoch 800: sigmoid = (0.016, 0.017), tanh = (0.008, 0.012), lrelu = (0.007, 0.012)

sigmoid (4)
abnormal sample: 38400, normal sample: 4800
(length    1) auc = 0.720
(length  120) auc = 0.839
(length 1200) auc = 0.844

sigmoid + drop (4)
abnormal sample: 38400, normal sample: 4800
(length    1) auc = 0.725
(length  120) auc = 0.843
(length 1200) auc = 0.859

tanh (4)
abnormal sample: 38400, normal sample: 4800
(length    1) auc = 0.781
(length  120) auc = 0.919
(length 1200) auc = 0.953

tanh + drop (4)
abnormal sample: 38400, normal sample: 4800
(length    1) auc = 0.787
(length  120) auc = 0.941
(length 1200) auc = 0.961

lrelu (4)
abnormal sample: 38400, normal sample: 4800
(length    1) auc = 0.785
(length  120) auc = 0.953
(length 1200) auc = 0.977

lrelu + drop (4)
abnormal sample: 38400, normal sample: 4800
(length    1) auc = 0.782
(length  120) auc = 0.961
(length 1200) auc = 0.992

combination (4)
abnormal sample: 38400, normal sample: 4800
(length    1) auc = 0.797
(length  120) auc = 0.953
(length 1200) auc = 0.977
```
