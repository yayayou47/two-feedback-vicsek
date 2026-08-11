"""Vérification complète des références d'un paquet LaTeX à deux fichiers.

Le manuscrit et le supplément se citent mutuellement via xr-hyper, ce qui
crée une classe d'erreurs que la compilation ne signale pas toujours :
une référence externe non résolue sort en « ?? » dans le PDF sans
provoquer d'erreur fatale, et un commit antérieur du dépôt porte
justement le message « fix broken SM cross-refs ».

Ce script vérifie six choses :

  1. citations \\cite dont la clé n'existe pas dans le .bib
  2. entrées du .bib jamais citées (avertissement)
  3. \\ref / \\eqref internes sans \\label correspondant
  4. références EXTERNES (xr-hyper) sans cible dans l'autre document
  5. « ?? » effectivement présents dans le PDF rendu
  6. fichiers de figures appelés mais absents du disque

Code de sortie non nul si l'une des catégories bloquantes échoue.

Lancer :  python3 submission_form/check_refs.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read_with_inputs(path: Path, seen: set[Path] | None = None) -> str:
    if seen is None:
        seen = set()
    path = path.resolve()
    if path in seen or not path.exists():
        return ""
    seen.add(path)
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        out.append(line)
        code = re.split(r"(?<!\\)%", line, maxsplit=1)[0]
        for m in re.finditer(r"\\(?:input|include)\s*\{([^}]*)\}", code):
            sub = m.group(1).strip()
            sub = sub if sub.endswith(".tex") else sub + ".tex"
            out.append(read_with_inputs(path.parent / sub, seen))
    return "\n".join(out)


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*$", "", s, flags=re.M)


def labels_of(tex: str) -> set[str]:
    return set(re.findall(r"\\label\{([^}]*)\}", tex))


def refs_of(tex: str) -> list[tuple[str, str]]:
    """(commande, cible) pour toutes les formes de renvoi."""
    out = []
    for cmd in ("ref", "eqref", "autoref", "cref", "Cref", "pageref"):
        for m in re.finditer(rf"\\{cmd}\{{([^}}]*)\}}", tex):
            for t in m.group(1).split(","):
                out.append((cmd, t.strip()))
    return out


def external_refs(tex: str, prefixes: list[str]) -> list[tuple[str, str]]:
    """Renvois vers l'autre document, repérés par le préfixe xr-hyper."""
    return [(c, t) for c, t in refs_of(tex)
            if any(t.startswith(p) for p in prefixes)]


def main() -> int:
    fail, warn = [], []

    docs = {n: ROOT / f"{n}.tex" for n in ("manuscript", "supplement")}
    docs = {n: p for n, p in docs.items() if p.exists()}
    if not docs:
        print("aucun .tex trouvé"); return 2

    texts = {n: strip_comments(read_with_inputs(p)) for n, p in docs.items()}

    # -- xr-hyper : \externaldocument{f} SANS préfixe est le cas courant.
    # Les labels de l'autre document sont alors indiscernables des labels
    # locaux, et un renvoi "orphelin" peut parfaitement être un renvoi
    # croisé valide. Il faut donc chercher la cible dans les DEUX
    # documents avant de conclure.
    prefixes = {}
    for n, t in texts.items():
        prefixes[n] = [p for p in
                       re.findall(r"\\externaldocument\[([^\]]*)\]", t) if p]

    bib_files = list(ROOT.glob("*.bib"))
    bib = "\n".join(f.read_text(encoding="utf-8", errors="replace")
                    for f in bib_files)
    defined = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))

    print("=" * 66)
    print("VÉRIFICATION DES RÉFÉRENCES")
    print("=" * 66)

    all_cited: set[str] = set()
    for n, t in texts.items():
        labels = labels_of(t)
        cited: set[str] = set()
        for m in re.finditer(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}", t):
            cited.update(k.strip() for k in m.group(1).split(",") if k.strip())
        all_cited |= cited

        other = "supplement" if n == "manuscript" else "manuscript"
        other_labels = labels_of(texts[other]) if other in texts else set()
        ext = set(external_refs(t, prefixes.get(n, [])))
        internal = [(c, tg) for c, tg in refs_of(t) if (c, tg) not in ext]
        # cible absente ICI mais présente dans l'autre document = renvoi
        # croisé valide, pas un orphelin
        cross = sorted({tg for _, tg in internal
                        if tg not in labels and tg in other_labels})
        dangling = sorted({tg for _, tg in internal
                           if tg not in labels and tg not in other_labels})

        print(f"\n--- {n}.tex ---")
        print(f"  labels définis        : {len(labels)}")
        print(f"  renvois internes      : {len(internal)}")
        print(f"  renvois croisés (xr)  : {len(cross)} vers {other}.tex")
        print(f"  citations             : {len(cited)}")

        if dangling:
            fail.append(f"{n}: renvois internes sans \\label : "
                        f"{', '.join(dangling[:8])}")
        missing_cites = sorted(cited - defined)
        if missing_cites:
            fail.append(f"{n}: citations absentes du .bib : "
                        f"{', '.join(missing_cites[:8])}")

        # collision de labels : avec un \externaldocument sans préfixe,
        # un même label défini des deux côtés se résout silencieusement
        # vers le mauvais objet. C'est le piège de cette configuration.
        clash = sorted(labels & other_labels)
        if clash:
            fail.append(f"{n}: label(s) définis dans les DEUX documents, "
                        f"renvoi ambigu : {', '.join(clash[:8])}")

        if other in texts and ext:
            bad = []
            for cmd, tg in sorted(ext):
                for p in prefixes.get(n, []):
                    if tg.startswith(p):
                        if tg[len(p):] not in other_labels:
                            bad.append(tg)
            if bad:
                fail.append(f"{n}: renvois xr sans cible dans "
                            f"{other}.tex : {', '.join(sorted(set(bad))[:8])}")

    orphans = sorted(defined - all_cited)
    if orphans:
        warn.append(f"{len(orphans)} entrée(s) du .bib jamais citée(s) : "
                    f"{', '.join(orphans[:6])}"
                    + (" …" if len(orphans) > 6 else ""))

    # -- « ?? » dans les PDF rendus
    print("\n--- PDF rendus ---")
    for n in docs:
        pdf = ROOT / f"{n}.pdf"
        if not pdf.exists():
            warn.append(f"{n}.pdf absent, contrôle du rendu sauté")
            continue
        try:
            txt = subprocess.run(["pdftotext", str(pdf), "-"],
                                 capture_output=True, text=True,
                                 timeout=90).stdout
        except Exception:
            warn.append(f"pdftotext indisponible pour {n}.pdf")
            continue
        qq = len(re.findall(r"(?<![?\w])\?\?(?![?\w])", txt))
        print(f"  {n}.pdf : {len(txt.split()):>6} mots, "
              f"{qq} occurrence(s) de « ?? »")
        if qq:
            fail.append(f"{n}.pdf contient {qq} renvoi(s) non résolu(s) "
                        f"affiché(s) « ?? »")

    # -- figures appelées mais absentes
    print("\n--- figures ---")
    missing_fig = []
    for n, t in texts.items():
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", t):
            g = m.group(1).strip()
            cands = [ROOT / g] + [ROOT / f"{g}{e}" for e in
                                  (".pdf", ".png", ".jpg", ".eps")]
            cands += [ROOT / "figures" / g] + [ROOT / "figures" / f"{g}{e}"
                                               for e in (".pdf", ".png",
                                                         ".jpg", ".eps")]
            if not any(c.exists() for c in cands):
                missing_fig.append(f"{n}: {g}")
    print(f"  appels \\includegraphics : "
          f"{sum(len(re.findall(r'includegraphics', t)) for t in texts.values())}")
    if missing_fig:
        fail.append("figures introuvables : " + ", ".join(missing_fig[:6]))
    else:
        print("  toutes les figures appelées existent")

    print("\n" + "=" * 66)
    for w in warn:
        print(f"avertissement : {w}")
    if fail:
        print(f"\nÉCHEC ({len(fail)}) :")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("OK : citations, renvois internes, renvois croisés, rendu et "
          "figures\n     sont tous résolus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
