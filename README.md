# Fletcher

While sequence alignments have historically helped infer activity once homology is identified, this is usually limited by the ability of the MSA tool to spot similarities and align them correctly. With the advent of AlphaFold, structural similarities may be found based on sequence similarities that would have flown under the radar if using an MSA exclusively. 'Fletcher' is a tool that will get a list of residues (and alternatives) and look for them in AlphaFold models provided that they lie within a fixed distance, calculated from the last atom of the first residue in a list of candidate residues. 

Usage: 

```
fletcher.py [-h] -f FILENAME -r RESIDUES -d DISTANCE

Fletcher will try to find a list of residues within a fixed distance from the last atom in the first residue.
Concept: Federico Sabbadin & Jon Agirre, University of York, UK.
Code: Jon Agirre, with contributions from Rebecca Taylor, University of York, UK.
Latest source code: https://github.com/glycojones/fletcher

Required arguments:

  -f FILENAME, --filename FILENAME
                        The name of the file to be processed, in PDB or mmCIF
                        format
  -r RESIDUES, --residues RESIDUES
                        A list of residues in one-letter code, comma
                        separated. Alternatives separated by ~, rotamers separated by ':'
                        e.g. H:3,H:3,W~F~Y:3 – two histidines in rotamer form 3, plus a
                        tryptophan, phenylalanine (any rotamer) or tyrosine in rotamer 3.
  -d DISTANCE, --distance DISTANCE
                        Specifies how far each of the residues can be from the
                        last atom (PDB order) in the first specified residue, in Angstroems
  -p PLDDT, --plddt PLDDT
                        Flag up candidate residues with average pLDDT below
                        thresold (Jumper et al., 2020).

Optional arguments:

-h, --help            show this help message and exit
```

Fletcher is not an acronym. It is the surname of the greatest musical catalyst I know: Guy Fletcher (https://www.guyfletcher.co.uk). 
