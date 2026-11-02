# Dependencies and research inputs

The repository implements research experiments around SplitFedV2 and NA-SFL. The
current source imports the following projects; their code, binaries, datasets,
and licenses remain governed by their respective maintainers' terms.

- [PyTorch](https://github.com/pytorch/pytorch) and
  [torchvision](https://github.com/pytorch/vision): tensors, autograd, ResNet
  BasicBlock/ResNet classes, CIFAR loading and transformations. This repository
  configures torchvision's ResNet18 with a CIFAR stem and ten output classes.
- [Mininet](https://github.com/mininet/mininet), pinned at
  `6eb8973c0bfd13c25c244a3871130c5e36b5fbd7`: Linux namespaces, hosts,
  TCLink and LinuxBridge. The Dockerfile installs its upstream source; Mininet
  source is not vendored here.
- [Open MPI](https://www.open-mpi.org/) and [mpi4py](https://github.com/mpi4py/mpi4py):
  the existing entrypoint's communicator import. Current learning runs use one
  process, not distributed MPI ranks.
- NumPy, Pillow, PyYAML, matplotlib and pytest: data processing, configuration,
  reporting and CPU tests. Exact active versions are in `requirements.txt`.
- [NVIDIA CUDA container images](https://hub.docker.com/r/nvidia/cuda) and CUDA
  runtime wheels: the new-server build and GPU runtime. The original local
  research image is not published by this repository.
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html): Alex Krizhevsky,
  Vinod Nair and Geoffrey Hinton. Cite the dataset's technical report when using
  it for research. Dataset images, archives and pretrained weights are not
  redistributed here.

For detailed current method definitions and implementation assumptions, see
`docs/11-02/04-architecture/README.md`. Historical experimental claims should not
be interpreted as an exact reproduction of an original paper or as a guarantee
of performance on a different GPU/server.
