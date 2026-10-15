# MDA-family methods relevant to the transition-zone study

This is a targeted taxonomy of the main ensemble-smoother variants relevant to
the current implementation, not a claim that every application-specific
variant has been catalogued. Metadata and abstracts were checked for the
references below.

## Method families

| Family | Main distinction | Fit to this project |
|---|---|---|
| ES-MDA | Repeated ensemble-smoother updates with inflated observation covariance; stochastic implementations perturb observations, with inverse inflation factors summing to one. | This is the current 0D/1D update family. The code uses perturbed observations and a fixed geometric alpha schedule. The final likelihood receives the unused weight. |
| Adaptive ES-MDA | Choose inflation factors from a discrepancy principle or an adaptive rule rather than fixing the whole schedule in advance. | Directly relevant to the observed 0D/1D budget overshoot. It is a schedule variant that can preserve a desired final likelihood weight without changing the solver or prior. |
| Iterative ensemble smoothers (IES / EnRML) | Repeatedly relinearize and update against the same data, commonly with ensemble-space sensitivities and prior regularization. Variants differ in stochastic/deterministic treatment, parameterization, and globalization. | A meaningful alternative to ES-MDA, but not presently a switch in this code. Ensemble-space sensitivities may be fragile here because the transition windows have limited ensemble rank and the 2D operator is expensive. |
| LM-IES / regularized LM-MAC | Iterative smoother updates viewed as regularized Levenberg–Marquardt steps; damping or line-search controls nonlinear steps. | Related in spirit, but the current `lm_laplace.py` is an adjoint MAP optimizer followed by Laplace sampling, not an ensemble LM smoother. It already provides actual-forward acceptance and convergence controls for the final stage. |
| Deterministic square-root / transform smoothers | Replace random perturbed-observation updates with a deterministic ensemble transform, reducing update sampling noise. | Not implemented by the current `approx_update` call path. Worth a separate implementation only after the current budget/prior/data-weight search is stable. |
| Annealed-importance / tempered IES | Couple iterative smoother updates to a tempering or importance-sampling scheme to better represent non-Gaussian posteriors. | A larger methodological change; relevant if a Gaussian MAP/Laplace approximation remains inadequate, but too broad to mix into the current budget and variance tests. |
| Hybrid smoother workflows | Combine stochastic ES-MDA / IES stages, hierarchical or transformed priors, and/or a final optimization/sampling step. | Closest architectural match: multi-fidelity 0D/1D ES-MDA followed by a gated 2D MAP/Laplace stage. Fidelity choice is an additional axis, not itself an MDA algorithm. |

Other orthogonal axes include adaptive inflation schedules, covariance
localization, explicit model-error covariance, hierarchical prior estimation,
non-Gaussian transforms, and selective/weighted observation channels. These
should be treated as controlled design choices rather than indiscriminately
stacked “MDA variants.”

## Current-controller observations

- The 0D and 1D stages use a geometric alpha schedule (initial alpha 10, then
  multiply alpha by 10/9). For the current initial sequence this spends
  0.100 + 0.090 + 0.081 = **0.271** before crossing a nominal 0.25 stage cap.
  Therefore a `one_d_budget` of 0.25 leaves no ES-MDA increment for 1D. The
  observed behavior motivates the 0.32 experiment: it admits exactly the next
  increment (0.0729), then stops at cumulative weight 0.3439.
- The current final stage gates on the best pre-final 1D member's measurement
  misfit. When selected, 2D LM optimizes the MAP and Laplace sampling produces
  the replacement ensemble. This is distinct from applying an ensemble-based
  IES update at 2D.
- Observation reweighting must be assessed on the fixed original covariance
  and against Rh/Rv truth RMSE. Otherwise, a variance change can appear to
  improve the training misfit merely by changing the objective scale.
- Keep TV as a separately controlled regularization axis. The current
  LM/Laplace mode requires Gaussian prior and TV off, so TV comparisons should
  use a compatible branch or a separately declared endpoint rather than
  silently changing the MAP/Laplace objective.

## References

1. A. A. Emerick and A. C. Reynolds, “Ensemble smoother with multiple data
   assimilation,” *Computers & Geosciences*, vol. 55, pp. 3–15, 2013.
   https://doi.org/10.1016/j.cageo.2012.03.011
2. Y. Chen and D. S. Oliver, “Levenberg–Marquardt forms of the iterative
   ensemble smoother for efficient history matching and uncertainty
   quantification,” *Computational Geosciences*, vol. 17, pp. 689–703, 2013.
   https://doi.org/10.1007/s10596-013-9351-5
3. X. Luo, A. S. Stordal, R. J. Lorentzen, and G. Nævdal, “Iterative ensemble
   smoother as an approximate solution to a regularized minimum-average-cost
   problem: theory and applications,” *SPE Journal*, vol. 20, pp. 962–982,
   2015. https://doi.org/10.2118/176023-PA
4. P. N. Raanes, A. S. Stordal, and G. Evensen, “Revising the stochastic
   iterative ensemble smoother,” *Nonlinear Processes in Geophysics*, vol. 26,
   pp. 325–344, 2019. https://doi.org/10.5194/npg-26-325-2019
5. D. S. Oliver, “Hybrid iterative ensemble smoother for history matching of
   hierarchical models,” arXiv:2206.01116, 2022.
   https://arxiv.org/abs/2206.01116
6. G. Evensen, “Analysis of iterative ensemble smoothers for solving inverse
   problems,”
    *Computational Geosciences*, vol. 22, pp. 885–908, 2018.
    https://doi.org/10.1007/s10596-018-9731-y
7. J. Rafiee and A. C. Reynolds, “Theoretical and efficient practical
   procedures for the generation of inflation factors for ES-MDA,”
   *Inverse Problems*, vol. 33, no. 11, 115003, 2017.
   https://doi.org/10.1088/1361-6420/aa8cb2
8. A. S. Stordal and A. H. Elsheikh, “Iterative ensemble smoothers in the
   annealed importance sampling framework,” *Advances in Water Resources*,
   vol. 86, pp. 231–239, 2015.
   https://doi.org/10.1016/j.advwatres.2015.09.030
9. M. Bocquet and P. Sakov, “An iterative ensemble Kalman smoother,”
   *Quarterly Journal of the Royal Meteorological Society*, vol. 140,
   pp. 1521–1535, 2014. https://doi.org/10.1002/qj.2236
