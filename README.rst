inknest
=======

An implementation of the DNS algorithm described in  `Brewer (2010) <https://arxiv.org/abs/0912.2380v3>`_.
The goal is to manage to use the *stretch-move-based* invariant ensemble sampler `(Goodman & Weare, 2010) <https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-p.pdf>`_ because of its ability to outperform MH-based algorithm (especially in high dimensions) and its capacity to manage discontinuos pdfs.

In a nutshell the purpose of this code is the management of *black-box*  user-defined problems, minimizing:

  * the tuning necessary to get a proper run
  * (hopefully) the computation time

Other documentation I used is:

  * `Andrieu & Thoms <https://people.eecs.berkeley.edu/~jordan/sail/readings/andrieu-thoms.pdf>`_
  * `Neal (2000) <https://arxiv.org/abs/physics/0009028>`_
  * `Walter (2015) <https://arxiv.org/pdf/1412.6368.pdf>`_

and obviously

`Skilling (2006) <https://projecteuclid.org/journals/bayesian-analysis/volume-1/issue-4/Nested-sampling-for-general-Bayesian-computation/10.1214/06-BA127.short>`_



