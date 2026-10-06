# Buildup simulation inputs

PyECLOUD reads its simulation settings from Python-style `.input` files.
The simulation input file selects the machine, beam, and secondary emission
input files. Additional cloud files configure simulations with multiple clouds.

```{toctree}
:maxdepth: 1

simulation
machine
beam
secondary-emission
multiple-clouds
```

These pages migrate the input sections of the existing reference manual in
`doc/reference/src/reference.tex`, by Giovanni Iadarola, Eleonora Belli,
Philipp Dijkstal, Lotta Mether, Annalisa Romano, Giovanni Rumolo, and Eric Wulff.
The descriptions and stated defaults come from that manual and have not yet
undergone a full audit against the current implementation.

The {download}`original reference manual <../../doc/reference/reference.pdf>`
also describes simulation outputs and remains available during the migration.
