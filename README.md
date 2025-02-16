# EvoP 

# Environment
```bash
conda create -n skip python=3.10
git submodule update --init
pip install -r requirements.txt
cd lib/scikit-opt && pip install -e . && cd ../..
cd lib/transformers && pip install -e . && cd ../..
cd src/lm-evaluation-harness && pip install -e . && cd ../..
```
