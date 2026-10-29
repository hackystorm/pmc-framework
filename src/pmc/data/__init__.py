# SPDX-License-Identifier: GPL-3.0-only

"""Dataset parsing and corpus handling for the PMC framework.

Stdlib only. Content never enters the ledger: every admitted record carries
a digest and provenance, and the bytes stay in the instance's private store.
"""