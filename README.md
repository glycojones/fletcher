# Fletcher

While Multiple Sequence Alignments (MSA) may infer function once homology is identified, traditional MSA tools struggle when sequence identity drops, frequently missing similarities in proteins that share as little as 5% sequence identity, yet retain very similar folds and catalytic activity! With the advent of AlphaFold and other really powerful fold prediction software, structural similarities may now be detected directly from predicted 3D models, revealing functional relationships that would fly under the radar of sequence-based methods. Fletcher is a tool that takes a user-defined list of candidate residues (including alternative options) and searches AlphaFold models for spatial matches within a fixed distance threshold, measured from the C-alpha of the primary reference residue.

Usage: 

```
fletcher_maximus.py [-h] -f FILENAME -r RESIDUES -d DISTANCE

Fletcher will try to find a list of residues within a fixed distance from the last atom in the first
residue.
Concept: Federico Sabbadin & Jon Agirre, University of York, UK. 
Code: Tom Robinson, Rebecca Taylor, Sophie Birrell & Jon Agirre, University of York, UK. 
Latest source code: https://github.com/glycojones/fletcher

Multi-motif mode (recommended) required arguments:

  -f FILENAME, --filename FILENAME
                        The name of the file to be processed, in mmCIF (preferred) or PDB 
                        format.
  -m , --motifs MOTIFS
                        Multiple motifs separated by '|'. Each motif contains a list of residues in
                        one-letter code. Use ',' for AND between residues, and '~' for OR between
                        residues.
                        Can also specify rotamers as integers (check what rotamer you want using an
                        experimental structure first!).
                        Each motif: anchor,targets:distance.
                        Example: --motifs "H:3,A,F:3:6.0 | H~D"
                        Which means a histidine in rotamer 3, an alanine and a phenylalanine in
                        rotamer 3 with all residues' last atom (PDB order) within 6 Å, with another
                        histidine or aspartic acid, separated by PROXIMITY.
  --proximity PROXIMITY
                        Maximum allowed distance (Å) between the last atom (PDB order) of any two
                        residues from different motifs.
  -p PLDDT, --plddt PLDDT
                        Flag up candidate residues with average pLDDT below
                        threshold (Jumper et al., 2020).

Single-motif mode required arguments:

  -f FILENAME, --filename FILENAME
                        The name of the file to be processed, in mmCIF (preferred) or PDB format.
  -r RESIDUES, --residues RESIDUES
                        A list of residues in one-letter code, comma separated, including
                        alternatives, e.g. H:1,H:2,F:1~W:3:2.
  -d DISTANCE, --distance DISTANCE
                        Specifies how far the last atom (PDB order) in each of the residues can be
                        from the last atom in the first specified residue, in Angstroms.
  -p PLDDT, --plddt PLDDT
                        Flag up candidate residues with average pLDDT below
                        threshold (Jumper et al., 2020).

Optional arguments:

-h, --help            show this help message and exit
```

Fletcher is not an acronym. It is the surname of the greatest musical catalyst I know: Guy Fletcher (https://www.guyfletcher.co.uk). 
