# jaxrl2

If you use JAXRL2 in your work, please cite this repository in publications:
```
@misc{jaxrl,
  author = {Kostrikov, Ilya},
  doi = {10.5281/zenodo.5535154},
  month = {10},
  title = {{JAXRL: Implementations of Reinforcement Learning algorithms in JAX}},
  url = {https://github.com/ikostrikov/jaxrl2},
  year = {2022},
  note = {v2}
}
```

## Installation

Run
```bash
pip install --upgrade pip

pip install -e .
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
```

See instructions for other versions of CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

## Examples

[Here.](examples/)

## Tests

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= pytest tests
```

# Acknowledgements 

Thanks to [@evgenii-nikishin](https://github.com/evgenii-nikishin) for helping with JAX. And [@dibyaghosh](https://github.com/dibyaghosh) for helping with vmapped ensembles.