# Debiased Causal Tree: Heterogeneous Treatment Effects Estimation with Unmeasured Confounding
This is the python implementation of GBCT in the NeurIPS'22 paper: Debiased Causal Tree: Heterogeneous Treatment Effects Estimation with Unmeasured Confounding.

We proposed a tree-based heterogeneous treatment effects estimation in the presence of unmeasured confounding using observational data and historical controls. In this work, we consider the case where covariates and the outcome are collected at multiple timestamps, before and after the treatment. We take two timestamps $t\in \{t_1,t_2\} (t_1<t_2)$, as an example, where the treatment is imposed at the time $t_2$. Following the potential outcome framework (Rubin, 1978), denote by $Y_{t}=D Y^{(1)}_t + (1-D)Y^{(0)}_t$ the observed outcomes at the time $t$, where $Y^{(d)}_{t}$ is the potential outcome for the treatment $D=d\in\{0,1\}$. We remark that $Y_{t_1}=Y^{(0)}_{t_1}$ since the treatment does not take place at $t_1$. Let $X$ and $U$ be time-invariant covariates and unmeasured confounders, respectively. Our work is motivated by the following identity.
![catt-identity](./figures/catt_identity.png)


which holds regardless of $U$. In general, the confounding bias $\mathbf{b}_t$ does not vanish due to $U$. However, the smaller $\mathbf{b}_t$ is, the closer CATT could be to the difference in conditional expectations. We then propose a tree-based method, which, at high level, aims to find an optimal partition of the feature space such that the confounding bias is smallest within each small region.

## Installation
**Step #1: Prepare Environment**
```shell
# clone code
git clone [git addresss]
# update submodule
git submodule update --recursive --init
```

**Step #2: Requirements Libraries**
- python 3.7
- pyhocon==0.3.59
- numpy==1.17.5
- pandas==1.0.0
- cmake==3.22.1
- sklearn
- pybind11

or

```shell
pip install -r requirements.txt
```

**Step #3: Build**
```shell
# change directory
cd third_part/gbct_utils
# compile the gbct_utils.so
python setup.py build 
# you should determine the name of the .so file based on the actual situation.
mv build/lib.linux-x86_64-3.7/gbct_utils.cpython-37m-x86_64-linux-gnu.so ../../src/
```

Hint: *Our code has been successfully tested on the CentOS 7 operating system using both GCC 4.8.5 and 6.5.1.* 

## Train
- shell
``` shell
python  src/boosting.py -c conf_path -o model_name -d data_path -e coefficient
```


- api
```python
from pyhocon import ConfigTree, ConfigFactory
from bin import BinMapper
from gradient_did_tree import GradientDebiasedCausalTree
from boosting import Boosting

# read data ...
data = ...
valid_data = ...
gbct = Boosting(GradientDebiasedCausalTree, conf, bin_mapper)
gbct.fit(data, valid_data)
```

## Citation
```latex
@inproceedings{tang2022debiased,
  title={Debiased Causal Tree: Heterogeneous Treatment Effects Estimation with Unmeasured Confounding},
  author={Tang, Caizhi and Wang, Huiyuan and Li, Xinyu and Cui, Qing and Zhang, Ya-Lin and Zhu, Feng and Li, Longfei and Zhou, Jun and Jiang, Linbo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```