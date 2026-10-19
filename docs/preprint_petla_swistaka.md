# Pętla Świstaka: testbed pętli czasowej do badania uczenia się agentów AI
## Szkielet preprintu + treść preregistracji (v0.1, szkic do rozbudowy)

**Roboczy tytuł EN:** *Groundhog Loops: A Time-Loop Testbed for Measuring Learning Speed and Tactic Formation in AI Agents*
**Proponowane kategorie arXiv:** cs.AI, cs.MA (ew. cs.LG)
**Powiązane dokumenty:** `docs/projekt_petla_swistaka.md` (DG v1.2), `docs/badanie_elastycznosci_strategii.md` (moduł elastyczności v1.2), `docs/wdrozenie_serwera_agenta.md`; kod: `serwer_agenta/`.

---

## Streszczenie (szkic)

Przedstawiamy **Pętlę Świstaka** — środowisko testowe, w którym błąd agenta powoduje reset do checkpointu sprzed zadania, lecz agent zachowuje trwałą, ustrukturyzowaną pamięć pętli (licznik iteracji, skumulowany czas, rekordy porażek, własne hipotezy), a środowisko może — wyłącznie przy wykrytym zacieśnieniu strategii — minimalnie zmutować warunki z gwarancją unieważnienia dotychczasowej strategii (G1b). W przeciwieństwie do linii Reflexion/ExpeL (pamięć werbalna bez resetu motywacyjnego) oraz PAIRED/UED (mutacje maksymalizujące trudność), pętla jest tu **kontrolowanym instrumentem motywacyjnym**, a metryki elastyczności (SR_swobodne, PSR, TTS) pozwalają zmierzyć prędkość uczenia się i jakość formułowania taktyk niezależnie od zapamiętania.

## 1. Tezy nowości (claim do zaklepania)

- **C1.** Pętla czasowa jako *kontrolowany instrument motywacyjny* z pełną świadomością skumulowanego czasu po stronie agenta (vs. reset jako neutralny mechanizm RL).
- **C2.** Mutacje *minimalne i celowane*: wyzwalane mierzalnym lock-inem (reguły R0–R4), z gwarancją unieważnienia zablokowanej strategii (G1b) i gwarancją rozwiązywalności (G1) — w opozycji do mutacji maksymalizujących regret (PAIRED).
- **C3.** Wariant z *ukrytym celem i niepewnym efektywnym budżetem pętli* jako test inferencji celu i prędkości formułowania taktyk (luka literaturowa).
- **C4.** Konstrukty behawioralne (wyuczona bezradność, anchoring, hedging, churning) operacjonalizowane metrykami agentowymi (SR/PSR/TTS/H(S)/SI) z rozróżnieniem elastyczności i pozornej zmienności.

## 2. Metoda (skrót do sekcji preprintu)

Meta-proces M⁺ = (M, C, Φ, Ψ, V) nad POMDP; checkpoint C = (stan, seed, czas logiczny); trwały meta-stan Φ_k = (k, τ, H_k, Q_k) izolowany od snapshotu; hierarchia mutacji 0–3; funkcja kosztu z P_fatal(i), P_end, P_early; harness FastAPI z kontraktem DG 8.3 i logiem JSONL 8.3a. Szczegóły: DG sekcje 2–8.

## 3. Blok preregistracji (do OSF)

**Pytania:** PR1 — które czynniki zmieniają skłonność do zmiany strategii; PR2 — czy skłonność do alternatyw rośnie wewnątrz epizodu; PR2b — czy rośnie między epizodami przy dozwolonym kanale pamięci (grupa A7); PR3 — transfer elastyczności na niewidziane rodziny mutacji.

**Hipotezy (kierunkowe, do kalibracji [START]):**
- H1: kara za powtórzenie klasy ma zależność odwróconego U na SR_swobodne (optimum w środku siatki).
- H2: bogatszy format Φ (lekcje > streszczenia > surowe trace) skraca TTS i podnosi PSR.
- H3: mutacje G1b podnoszą PSR po mutacji, ale zbyt częste mutacje obniżają SR_swobodne (bezradność).
- H4: wzrost SR bez wzrostu PSR oznacza churning (kryterium falsyfikacji „elastyczności").

**Metryki pierwotne:** SR_swobodne, PSR (vs. baseline post-mutacyjny), TTS, L2S. **Wtórne:** H(S), Novelty Rate, SI, M7/M8.
**Analiza:** GLMM binomiale z wagami (agent-replika jako efekt losowy, min. 10 poziomów), Cox z frailty dla TTS, korekta Holma, MDE z symulacji na wariancji pilotaża F2/F3.
**Grupy:** A0–A6 (DG) + A7–A14 (moduł). **Kryteria wykluczeń/cenzurowania:** wg DG 8.3e (censored_mutation_limit).

## 4. Artefakty i reprodukowalność

Repo: github.com/casual-art/Groudhog-Day (kod + dokumenty + rejestr eksperymentów YAML). DOI: do nadania przez Zenodo po pierwszym release. Tryb replay i seed pinning wg DG 4.1/8.3.

## 5. Plan prac (skrót)

F0 prototyp (gotowy, scaffold) → F2 kalibracja progów [START] → F3 pilotaż + ablacje → F5 ewaluacja końcowa na holdout → preprint v1 na arXiv.

---

### Ściąga operacyjna dla autorki

**Zenodo DOI (5 min):** zenodo.org → zaloguj przez GitHub → włącz repo Groudhog-Day → na GitHubu zrób *Release* (np. v0.1.0) → Zenodo nada DOI i zarchiwizuje snapshot.
**OSF:** osf.io → New Project → „Preregistration" → wklej sekcje 3 powyżej.
**arXiv:** potrzebne konto + ewentualnie endorsement dla cs.AI; PDF z tego szkieletu po rozbudowie sekcji 2–3.
