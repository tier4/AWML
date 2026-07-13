"""Cross-backend numerical verification. Import concrete submodules (``deployment.verification.backend_verifier``, …).

Verification is a peer stage to evaluation (see ``deployment/runtime/verification_orchestrator.py``):
it compares one backend's outputs against another's via ``OutputComparator`` rather than scoring
metrics. It consumes the shared ``deployment.execution`` primitives.
"""
