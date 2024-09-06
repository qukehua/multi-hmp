
**Relation Learning and Aggregate-attention for
Multi-person Motion Prediction** 

### Abstract
------
Multi-person motion prediction is an emerging and
intricate task with broad real-world applications. Previous methods
achieve impressive results by modeling the spatial relations of
all individuals’ joints and the temporal trajectory of a particular
individual. However, they may ignore the distinction between
the spatial relationships within a person and those between
multiple persons, which inevitably introduces undesired dependencies.
To address this issue, we introduce a new collaborative
framework for multi-person motion prediction that explicitly
models these relations: a GCN-based network for intra-relations
(spatio-temporal relations inside a person) and a novel reasoning
network for inter-relations (spatial relations between persons).
Moreover, we propose a novel plug-and-play aggregation module
called the Interaction Aggregation Module (IAM), which employs
an aggregate-attention mechanism to seamlessly integrate these
relations. Experiments indicate that the module can also be
applied to other dual-path models. Extensive experiments on
the 3DPW, 3DPW-RC, CMU-Mocap, MuPoTS-3D, as well as
synthesized datasets Mix1 & Mix2 (9∼15 persons), demonstrate
that our method achieves state-of-the-art performance.

### Network Architecture
------
![image](images/architecture.png)

### Requirements
------
- PyTorch = 1.8.0
- Numpy
- CUDA = 11.4
- Python = 3.1.0

### Data Preparation
------
Download all the data and put them in the [dataset path].

[H3.6M](https://drive.google.com/file/d/15OAOUrva1S-C_BV8UgPORcwmWG2ul4Rk/view?usp=share_link)

Directory structure: 
```shell script
[dataset path]
|-- h36m
|   |-- S1
|   |-- S5
|   |-- S6
|   |-- ...
|   |-- S11
```


[CMU mocap](http://mocap.cs.cmu.edu/) 

[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)

Directory structure: 
```shell script
[dataset path]
|-- 3dpw
|   |-- sequenceFiles
|   |   |-- test
|   |   |-- train
|   |   |-- validation
```

### Training
------
+ Train on Human3.6M:

`
python main_h36m.py
--data_dir
[dataset path]
--num_gcn
4
--dct_n
15
--input_n
10
--output_n
10
--skip_rate
1
--batch_size
32
--test_batch_size
64
--node_n
66
--cuda_idx
cuda:0
--d_model
16
--lr_now
0.0005
--epoch
100
--test_sample_num
-1
`

+ Train on CMU-MoCap:

`
python main_cmu_3d.py
--data_dir
[dataset path]
--num_gcn
4
--dct_n
15
--input_n
10
--output_n
25
--skip_rate
1
--batch_size
16
--test_batch_size
32
--node_n
75
--cuda_idx
cuda:0
--d_model
16
--lr_now
0.005
--epoch
100
--test_sample_num
-1
`

+ Train on 3DPW:

`
--data_dir
[dataset path]
--num_gcn
4
--dct_n
15
--input_n
10
--output_n
30
--skip_rate
1
--batch_size
32
--test_batch_size
64
--node_n
69
--cuda_idx
cuda:0
--d_model
16
--lr_now
0.001
--epoch
100
--test_sample_num
-1
`



## Evaluation
------
Add `--is_eval` after the above training commands.

The test result will be saved in `./checkpoint/`.

#### Ackowlegments
Our code is based on [PGBIG](https://github.com/705062791/PGBIG) and [Dpnet](https://ieeexplore.ieee.org/document/10025861).
