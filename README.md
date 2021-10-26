# ensnest

[![docs - passing](https://img.shields.io/static/v1?label=docs&message=passing&color=2ea44f)](https://djanloo.github.io/ensnest/)
![os](https://img.shields.io/static/v1?label=os&message=Linux&color=2ea44f)

An ensemble nested sampling implementation using the *stretch-move-based* affine invariant ensemble sampler [(Goodman & Weare, 2010)](https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-p.pdf)

> **WARNING**: Due to multiprocessing issues v0.1-alpha works only under unix-based OSs.


## Notable features
- low tuning necessary
- multimodal is ok
- multiprocessed
- lots of progress bars

## Installation

Install with

```console
$ sudo python3 setup.py install
```

## Examples

### Eggbox model

(true) logZ = 235.88

(computed) logZ = 235.91 +- 0.05

in 37s

![eggbox](assets/presentation/eggbox.png)


### Rosenbrock function

(true) logZ = -2.17

(computed) logZ = -2.16 +- 0.02

in 25s

![eggbox](assets/presentation/rosenbrock.png)


### Gaussian shells

(true) logZ = -2.47

(computed) logZ = -2.44 +- 0.03

in 62s

![eggbox](assets/presentation/gaussianshells.png)
