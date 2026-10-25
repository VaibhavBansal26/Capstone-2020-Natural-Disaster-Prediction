"""`python -m ndp.evals ...` is the same as `ndp-eval ...`."""

import sys

from ndp.evals import main

raise SystemExit(main(sys.argv[1:]))
