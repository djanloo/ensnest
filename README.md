# inknest

An ensemble nested sampling implementation using the *stretch-move-based* affine invariant ensemble sampler [(Goodman & Weare, 2010)](https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-p.pdf)

For an explaination of the algorithm see ``conceptual_notes.pdf``, for the code user guide see ``documentation\build\inknest.pdf``

> **WARNING**: Due to multiprocessing issues v1.0 works only under unix-based OSs.


## Notable features
- very low tuning necessary
- correct management of multimodality
- multiprocessed

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



