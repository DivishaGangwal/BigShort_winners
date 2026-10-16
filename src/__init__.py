"""BigShort round-one shared pipeline.

Thread caps are set here so they apply before any numerical library is
imported by a submodule (PLAN.md section 4).
"""
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"
