# Fletcher

While Multiple Sequence Alignments (MSA) may infer function once homology is identified, traditional MSA tools struggle when sequence identity drops, frequently missing similarities in proteins that share as little as 5% sequence identity, yet retain very similar folds and catalytic activity! With the advent of AlphaFold and other really powerful fold prediction software, structural similarities may now be detected directly from predicted 3D models, revealing functional relationships that would fly under the radar of sequence-based methods. Fletcher is a tool that takes a user-defined list of candidate residues (including alternative options) and searches AlphaFold models for spatial matches within a fixed distance threshold, measured from the C-alpha of the primary reference residue.

Usage: 

```
fletcher.py [-h] -f FILENAME -r RESIDUES -d DISTANCE

Fletcher will try to find a list of residues within a fixed distance from the last atom in the first residue.
Concept: Federico Sabbadin & Jon Agirre, University of York, UK.
Code: Jon Agirre, with contributions from Rebecca Taylor, University of York, UK.
Latest source code: https://github.com/glycojones/fletcher

Required arguments:

  -f FILENAME, --filename FILENAME
                        The name of the file to be processed, in mmCIF (preferred) or PDB 
                        format
  -m , --motifs MOTIFS
                        Multiple motifs separated by '|'. Each motif: anchor,targets:distance.
                        Use ',' for AND between positions, and '~' for OR between residues.
                        Can also specify rotamers as integers (check what rotamer you want using an
                        experimental structure first!)
                        Example: --motifs "H:3,F:5.0 | H~D:5.0"
                        Which means a histidine in rotamer 3 and a phenyl alanine within 5 Å,
                        with another histidine or aspartic acid within 5 Å, separated by DISTANCE.
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
