# Source du manuscrit — soumission JSTAT

```
make            # compile manuscrit + supplément, dans le bon ordre, puis vérifie
make archive    # produit jstat_upload.tar.gz, l'archive à téléverser
make clean
```

## L'archive à téléverser

`make archive` produit `jstat_upload.tar.gz`, qui contient **exactement**
ce que JSTAT demande et rien de plus :

```
manuscript.tex      le fichier maître, à la racine de l'archive
refs.bib
manuscript.bbl      exigé : « If you have used BibTeX, you must include the .bbl file »
supplement.aux      voir ci-dessous — c'est la pièce sans laquelle tout casse
figures/            les 11 figures appelées par le manuscrit, et elles seules
```

Le guide est explicite : « The archive should contain only the files
necessary for the compilation and the production of the pdf (no cover
letters, reports, etc.) ». Ni ce README, ni le Makefile, ni le supplément,
ni la lettre n'y sont.

## Pourquoi `supplement.aux` est indispensable

Le manuscrit déclare `\externaldocument{supplement}` : ses renvois vers
le matériel supplémentaire (Fig. S1, S2, …) sont résolus en lisant
`supplement.aux`.

Le serveur JSTAT ne compile que le fichier maître. Il ne compilera donc
jamais `supplement.tex`, et sans le `.aux` livré d'avance **onze renvois
sortent en « ?? » dans le PDF, sans la moindre erreur de compilation**.
La soumission « réussit » et le PDF que lisent l'éditeur et les
rapporteurs est cassé.

Vérifié : en décompressant l'archive dans un répertoire vierge et en ne
compilant que `manuscript.tex`, on obtient 23 pages, zéro « ?? », zéro
référence non résolue.

Corollaire : si vous modifiez `supplement.tex`, relancez `make` avant
`make archive`, sinon le `.aux` embarqué sera périmé.

## Un seul .tex dans l'archive, volontairement

JSTAT prévient : « If two or more files have the same .tex extension […]
you must specify the master file name in the appropriate field,
otherwise the submission will fail. » L'archive n'en contient qu'un, ce
qui supprime le risque. Le supplément se téléverse séparément, en PDF,
par le bouton « upload attachment » **après** la soumission principale.

## Conformité au style JSTAT

Références en système **numérique séquentiel**, dans l'ordre de citation
et non alphabétique, comme l'exige le guide : `natbib` en mode
`[numbers,sort&compress]` avec `unsrtnat`. Les mots-clés figurent après
l'abstract.
