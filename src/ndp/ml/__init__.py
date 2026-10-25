"""Training code and model registry. Models are trained once with `ndp-train`
and loaded from disk at request time; nothing trains inside an HTTP handler."""
