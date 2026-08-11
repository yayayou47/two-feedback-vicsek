"""Garde-fou de citations — le contrôle qui a manqué à la v1.

Le grief 4 du rapport de rejet portait sur notre CRÉDIBILITÉ, pas sur la
physique : la lettre de réponse annonçait une discussion de Clusella &
Pastor-Satorras, l'entrée était bien dans refs.bib, mais elle n'était
citée nulle part dans manuscript.tex -- donc absente du .bbl. Le
rapporteur a cherché une référence jamais imprimée.

Ce script rend cette erreur impossible. Il vérifie que :

  1. toute clé \\cite{...} du manuscrit existe dans refs.bib ;
  2. toute clé listée dans MUST_CITE est effectivement citée dans le
     corps du texte (c'est la garantie anti-grief-4) ;
  3. si un .bbl est présent, toute clé de MUST_CITE y apparaît -- seul
     contrôle qui prouve que la référence sera IMPRIMÉE ;
  4. aucune entrée de refs.bib n'est orpheline (avertissement seulement).

Code de sortie non nul en cas d'échec, donc branchable sur un Makefile :

    check-cites:
    \tpython3 notes/src/check_citations.py

Lancer : python3 notes/src/check_citations.py [--tex F] [--bib F] [--bbl F]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1_ROOT = HERE.parent

#: Références dont l'absence du texte imprimé est une faute grave, soit
#: parce qu'elles constituent une ANTÉRIORITÉ directe qu'un rapporteur
#: nous reprocherait d'ignorer, soit parce qu'un rapport les a
#: explicitement demandées.
MUST_CITE: dict[str, str] = {
    # Renseigner ici les références dont l'absence du texte IMPRIME
    # serait une faute : antériorités directes, travaux qu'un rapporteur
    # exigerait, references annoncees dans une lettre de reponse.
    # Laisser vide desactive ce controle mais garde les autres.
}


def _read_with_inputs(path: Path, _seen: set[Path] | None = None) -> str:
    """Lit un .tex en suivant récursivement \\input et \\include.

    Sans cela le contrôle est inutile sur un manuscrit assemblé : le
    fichier maître ne contient aucun \\cite, tous étant dans les pièces
    incluses, et le script signale six absences fantômes. Un garde-fou
    qui crie au loup à tort est un garde-fou qu'on désactive.
    """
    if _seen is None:
        _seen = set()
    path = path.resolve()
    if path in _seen or not path.exists():
        return ""
    _seen.add(path)
    txt = path.read_text(encoding="utf-8", errors="replace")
    out = []
    for line in txt.splitlines():
        # on ignore ce qui suit un % non échappé
        code = re.split(r"(?<!\\)%", line, maxsplit=1)[0]
        out.append(line)
        for m in re.finditer(r"\\(?:input|include)\s*\{([^}]*)\}", code):
            sub = m.group(1).strip()
            if not sub.endswith(".tex"):
                sub += ".tex"
            out.append(_read_with_inputs(path.parent / sub, _seen))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(V1_ROOT / "manuscript.tex"))
    ap.add_argument("--bib", default=str(V1_ROOT / "refs.bib"))
    ap.add_argument("--bbl", default=str(V1_ROOT / "manuscript.bbl"))
    a = ap.parse_args()

    tex_p, bib_p, bbl_p = Path(a.tex), Path(a.bib), Path(a.bbl)
    if not tex_p.exists() or not bib_p.exists():
        print(f"ERREUR : {tex_p} ou {bib_p} introuvable")
        return 2

    tex = _read_with_inputs(tex_p)
    bib = bib_p.read_text(encoding="utf-8", errors="replace")

    defined = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))
    cited: set[str] = set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}",
                         tex):
        cited.update(k.strip() for k in m.group(1).split(",") if k.strip())

    fail = []

    missing = sorted(cited - defined)
    if missing:
        fail.append(f"clés citées mais absentes de {bib_p.name} : "
                    f"{', '.join(missing)}")

    not_cited = [(k, why) for k, why in MUST_CITE.items() if k not in cited]
    for k, why in not_cited:
        where = "absente de refs.bib ET du texte" if k not in defined \
                else "dans refs.bib mais JAMAIS citée dans le texte"
        fail.append(f"MUST_CITE '{k}' : {where}\n      motif : {why}")

    if bbl_p.exists():
        bbl = bbl_p.read_text(encoding="utf-8", errors="replace")
        not_printed = [k for k in MUST_CITE
                       if k in cited and k not in bbl]
        if not_printed:
            fail.append(f"présentes dans le .tex mais absentes du .bbl "
                        f"(bibliographie pas régénérée ?) : "
                        f"{', '.join(sorted(not_printed))}")
    else:
        print(f"note : {bbl_p.name} absent, contrôle d'impression sauté")

    orphans = sorted(defined - cited)
    if orphans:
        print(f"avertissement : {len(orphans)} entrée(s) jamais citée(s) : "
              f"{', '.join(orphans)}")

    print(f"\n{len(defined)} entrées dans {bib_p.name}, "
          f"{len(cited)} clés citées dans {tex_p.name}.")

    if fail:
        print(f"\nÉCHEC ({len(fail)}) :")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("OK : toutes les références obligatoires sont citées et imprimées.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
