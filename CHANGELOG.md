# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- Small tweak to QR decomposition lapack call -> overall runtime and memory usage improvement of ~24% and 19%, respectively. Oops.

## [0.5.0] - 2022-08-07
### Added
- `test_vars` option to specify the variable(s) for which to calculate PL confidence intervals and p-values.
### Fixed
- Fixed bug where `.summary(xname)` would append `Intercept` to `xname` such that repeated calls would break.

## [0.4.0] - 2022-08-01
### Added
- Option to use Wald method for computing p-values and confidence intervals instead of LRT and profile likelihood. Set `wald=True` to use ([#11](https://github.com/jzluo/firthlogist/pull/11)).
- Tests for `load_sex2()` and `load_endometrial()` ([#9](https://github.com/jzluo/firthlogist/pull/9)).
- Test for profile likelihood confidence intervals ([#9](https://github.com/jzluo/firthlogist/pull/9)).
### Changed
- `skip_lrt` option is now `skip_pvals` ([#11](https://github.com/jzluo/firthlogist/pull/11)).
### Fixed
- `.summary()` no longer breaks if skipping confidence interval or p-value calculation ([#11](https://github.com/jzluo/firthlogist/pull/11)).
### Removed
- Diabetes and sex2 csv files removed from testing dir ([#9](https://github.com/jzluo/firthlogist/pull/9)).

## [0.3.1] - 2022-07-29
### Added
- Added the endometrial cancer dataset analyzed in Heinze and Schemper, 2002. Load using `load_endometrial()` ([#8](https://github.com/jzluo/firthlogist/pull/8)).
### Changed
- Disabled step-halving by default to follow `logistf`, which disabled it in version 1.24.1 for some reason ([#8](https://github.com/jzluo/firthlogist/pull/8)).

## [0.3.0] - 2022-07-28
v0.3.0 adds a couple of convenience features.
### Added
- Added `.summary()` method to print summary of results ([#6](https://github.com/jzluo/firthlogist/pull/6)). See the readme for a usage example.
- Added the sex2 dataset from logistf as `load_sex2()` ([#7](https://github.com/jzluo/firthlogist/pull/7)).

## [0.2.0] - 2022-07-27
v0.2.0 is the MVP release implemented in Numpy and Scipy.
### Added
- Calculate confidence intervals using profile penalized likelihood ([#5](https://github.com/jzluo/firthlogist/pull/5)).
- `skip_ci` parameter to skip calculation of confidence intervals.
- `alpha` parameter to specify confidence interval.
- `pl_max_iter`, `pl_max_halfstep`, `pl_max_stepsize` parameters for controlling profile penalized likelihood procedure.
### Changed
- Predictions calculated during fitting procedure are now clipped to the range `[1e-15, 1-1e-15]` instead of `[0, 1]`.


## [0.1.2] - 2022-06-27
### Fixed
- Added missing `loglik_` attribute to readme.

## [0.1.1] - 2022-06-27
### Added
- Parameters and attributes documentation in readme.

## [0.1.0] - 2022-06-26
### Added
- Penalized likelihood ratio test implemented for calculating p-values ([#2](https://github.com/jzluo/firthlogist/pull/2)).
- `skip_lrt` parameter for skipping the p-value calculations.

## [0.0.4] - 2022-06-13
### Added
- Calculate coef standard errors in `_bse()`.
- `loglik_` and `_bse` attributes for fitted log-likelihood and coef standard errors.

## [0.0.3] - 2022-06-12
### Changed
- Call lapack routines to get Q only instead of `np.linalg.qr`.

## [0.0.2] - 2022-06-11
### Changed
- Added info for PyPI in pyproject.toml.
- Added install instruction to readme.

## [0.0.1] - 2022-06-11
Initial release
