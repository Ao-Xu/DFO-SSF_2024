## Reproduction of the DFO-S Algorithm

**This code is based on my paper:**

Xu A., Li Z., Zhang Y., et al. *Generating Image Counterfactuals in Deep Learning Models Without the Aid of Generative Models.* IEEE Signal Processing Letters, 2025.

### Overview

This repository provides the implementation to reproduce the DFO-S (Deep Learning-based Optimization for Counterfactual Explanations) algorithm. The key steps involved are:

1. Download and load the MNIST dataset.
2. Define a simple image classifier.
3. Define the scoring network \$s\_{\theta}(\cdot)\$.
4. Train a classifier model.
5. Train a scoring network for each class.
6. Compute the synthesized score field.
7. Implement DFO-S based on Algorithm 1 and Algorithm 2.

### Key Algorithms

* **Algorithm 1**: Use scoring synthesis techniques to identify the target class.
* **Algorithm 2**: DFO-S — Generate image counterfactuals under the synthesized score field constraint.

### Steps to Reproduce the DFO-S Algorithm

#### Environment Setup (Linux)

To set up the environment, execute the following commands:

```bash
conda create -n dfos python=3.10 -y
source ~/.bashrc
conda init bash
conda activate dfos
```

#### Install Required Packages

Install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

#### MNIST Dataset Implementation

The MNIST dataset can be loaded (or converted for other datasets) using this code:

```python
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
```

#### Define and Train the Scoring Network

* Run **Algorithm 1** to identify the target class.
* Run **Algorithm 2** to generate counterfactual images.

```bash
python3 dfo_s_mnist_main.py
```

*Note*: The initial version of the code has been updated, and these two modules are not fully compatible, so you may skip this step.

#### Visualization

When running on a Linux server (which typically lacks a GUI), visualizations using `plt.show()` may fail to display images. For generating image files instead, use this script:

```bash
python3 dfo_s_mnist_picture.py
```

## Code Availability

We are fully committed to reproducibility. This code is publicly available on GitHub and can be used to reproduce the results in the paper. You can access the repository and the required experimental setup directly from here.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite our paper:

```
@article{dfo-s2025,
  title={Generating Image Counterfactuals in Deep Learning Models Without the Aid of Generative Models},
  author={Xu, A. and Li, Z. and Zhang, Y. and et al.},
  journal={IEEE Signal Processing Letters},
  year={2025}
}
```
