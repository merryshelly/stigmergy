"""Worker-image assets (Stigmergy bead .63).

`shim.py` is a pure-stdlib module so the SAME file runs standalone inside the
container (`python3 /opt/stigmergy/shim.py egress`) AND imports on the host
for unit tests. Nothing here may import the `stigmergy` package — these files
are copied into a minimal container that does not have it installed.
"""
