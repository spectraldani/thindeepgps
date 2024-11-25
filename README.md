# Thin and Deep Gaussian Processes
This repository is the official implementation of the methods in the publication
* Daniel Augusto de Souza, Alexander Nikitin, S. T. John, Magnus Ross, Mauricio A. Álvarez, Marc Peter Deisenroth, João P. P. Gomes, Diego Mesquita, and César Lincoln Mattos. **Thin and Deep Gaussian Processes**. In *Advances in Neural Information Processing Systems (NeurIPS) 2023*.

The paper can be downloaded from [ArXiv](https://arxiv.org/abs/2310.11527) and we have also written a [blog post](https://spectral.space/pubs/Souza2023-tdgp.html) explaining the paper.

<p align="center">
  <img src="static/tdgp.png" />
</p>

# Install

This library was implemented against Python 3.10.8. To install it and its dependencies, please do:

```
pip install -r requirements.txt
pip install --no-deps uq360==0.2
python setup.py develop
```

# Run:
To reproduce the plots and results of our synthetic experiment, please run:
```
python 'synthetic experiment.py'
```
This script will generate the plots used in paper alongside additional plots. This script is also formatted as a percent format notebook that can also be open as a [jupytext](https://jupytext.readthedocs.io/en/latest/index.html) notebook or in many IDEs (Spyder, VS Code, PyCharm, etc).

## Citation
If you use the code in this repository for your research, please cite the paper as follows:
```bibtex
@InProceedings{Souza2023-tdgp,
    author = {de Souza, Daniel Augusto and Nikitin, Alexander and John, S. T. and Ross, Magnus and Álvarez, Mauricio A. and Deisenroth, Marc Peter and Gomes, João P. P. and Mesquita, Diego and Mattos, César Lincoln},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS) 36},
    date = {2023-12},
    title = {Thin and Deep {G}aussian Processes},
    volume = {36},
    url = {https://proceedings.neurips.cc/paper_files/paper/2023/hash/2aa212d6f40c1cb19b777e83db00ec6a-Abstract-Conference.html},
}
```
