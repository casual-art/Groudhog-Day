# Badanie elastyczności behawioralnej agenta w pętli czasowej — dokument towarzyszący do projektu „Pętla Świstaka"

**Status:** dokument badawczy towarzyszący, wersja 1.2 (po recenzji REVISE; dodano sekcję 11 „Kontekst literaturowy"; changelog w sekcji 10)
**Relacja do dokumentu głównego:** niniejszy dokument nie modyfikuje środowiska ani protokołu z dokumentu głównego (dalej: DG, wersja 1.2); definiuje warstwę pomiarową i eksperymentalną nakładaną na metryki M1–M8 oraz grupy A0–A6, rozszerzone o jawne nowe grupy ablacyjne A7–A14 (sekcja 6.1). Wszystkie wartości `[START]` podlegają kalibracji w fazie F2 (DG sekcja 12).
**Konwencja:** odwołania „DG X.Y" wskazują sekcje dokumentu głównego.

---

## 1. Streszczenie i pytania badawcze

Dokument główny koncentruje się na tym, *czy* agent uczy się z pętli (L2S, M1) i *czy* mutacje przerywają lock-in (C3). Niniejsze badanie przesuwa oś analizy na **elastyczność behawioralną**: gotowość agenta do porzucania klas strategii i sięgania po alternatywy, oraz na **meta-uczenie elastyczności** — zmianę tej gotowości wraz z doświadczeniem pętli.

**PR1 (czynniki switcha).** Które elementy systemu — struktura kosztów (α, β, P_fatal, P_end, P_early, kara za powtórzenie klasy), prezentacja pamięci Φ, polityka mutacji Ψ, widoczność liczników pętli, trafność diagnozy — zwiększają, a które zmniejszają prawdopodobieństwo zmiany klasy strategii po porażce? Hipoteza przewodnia: zależność większości czynników jest nieliniowa (odwrócone U), z optimum w strefie średniej presji.

**PR2 (trajektoria poprawy wewnątrzepizodowej).** Czy tendencja do stosowania alternatywnych rozwiązań **rośnie w obrębie epizodu** wraz z numerem pętli k i narastaniem porażek — tj. czy agent uczy się nie tylko konkretnego zadania, lecz reguły „po powtarzającej się porażce warto zmienić klasę"? Kryterium poprawy: dodatni trend SR_swobodne i PSR oraz ujemny trend TTS wzdłuż k w obrębie epizodu (definicje w sekcji 4, model w sekcji 5.1).

*Uwaga o identyfikowalności (zmiana względem v1.0).* W architekturze DG meta-stan Φ jest per epizod (kasowany między epizodami), wagi RL/LLM nie są aktualizowane w trakcie eksperymentu (DG 3.1, „poza zakresem" w DG 1), a prompt systemowy jest stały — zatem trend **między** epizodami nie ma kanału uczenia i **nie może być kryterium sukcesu**: zgodnie z DG 10.3 poprawa między epizodami jest traktowana jako naruszenie założeń (przeciek poza kanał Φ). Trend międzyepizodowy przekwalifikowujemy z kryterium sukcesu na **test ważności** (sekcja 5.2): istotny trend po epizodach jest alarmem o przecieku/cache, nie dowodem meta-uczenia.

**PR2b (meta-uczenie elastyczności przy dozwolonym kanale międzyepizodowym).** `[START — eksperyment rozszerzający]` Czy agent wyposażony w trwały kanał pamięci międzyepizodowej (nowa grupa ablacyjna **A7**, sekcje 5.3 i 6.1) uczy się *elastyczności jako reguły* — tj. czy SR_swobodne/PSR w pętli 0–1 rosną, a TTS maleje wraz z numerem epizodu, gdy kanał uczenia jest legalnie otwarty? PR2b jest badawczo odrębne od PR2: PR2 testuje uczenie w obrębie epizodu w niezmodyfikowanym protokole DG; PR2b testuje meta-uczenie w świadomie rozszerzonej architekturze.

**PR3 (transfer).** Czy nabyta elastyczność przenosi się na scenariusze holdout z rodzinami mutacji niewidzianymi w fazie kalibracji (DG 10.3), czy jest przywiązana do schematu Ψ (overfitting do mutacji, DG 11)? Kryterium: margines non-inferiority na skali absolutnej z przedziałem ufności, po dopasowaniu trudności pul dev/holdout (sekcja 5.4).

Wkład badania: operacjonalizacja switcha odporna na churning i na konfundację z mutacjami (SR_swobodne, PSR względem baseline'u post-mutacyjnego); taksonomia czynników z hipotezami kierunkowymi (w tym trafności diagnozy, F15); plan eksperymentów E1–E6 i grupy A7–A14 wpięte w harmonogram F2–F5, z tabelą budżetu i symulacyjną analizą mocy (MDE).

---

## 2. Definicje operacyjne

### 2.1. Klasa strategii i odległość między klasami

Przyjmujemy definicję klasy strategii z DG 2.3 (klasa równoważności trajektorii względem strat(σ)), z **jedną, ustaloną konkretyzacją per typ agenta**, zgodną z DG 6.2 i DG 10.1-M2:

- **Agent RL / gridworld (S1):** klasa = sekwencja odwiedzonych regionów. Identyczność klas = **dokładna równość sekwencji regionów**; podobieństwo klas = **odległość edycyjna (Levenshtein)** na sekwencjach regionów (jak w DG 10.1-M2). Embedding trajektorii z zamrożonego autoenkodera (DG 3.4) służy wyłącznie pomocniczo — do metryki głębokości SI — i nie rozstrzyga o tożsamości klas.
- **Agent LLM (S2, S3):** klasa = **znormalizowana sekwencja nazw narzędzi bez argumentów**. Identyczność klas = porównanie tekstowe znormalizowanych planów; przy ocenie ciągłej — cosinus na embeddingach planów z progiem **cos ≥ 0.9** `[START]` uznawanym za „tę samą klasę" (jak w DG 6.2).

**Odległość między klasami (do SI):** d(C_a, C_b) = 1 − cos(emb(C_a), emb(C_b)) ∈ [0, 2]. Klasy o cos ≥ 0.9 traktujemy jako tę samą klasę; skuteczny zakres metryki SI wynosi zatem **(0.1, 2]** (poprawka względem v1.0: cosinus bywa ujemny, więc górna granica to 2, nie 1). Switch jest tym głębszy, im większe d (sekcja 4, metryka SI). Próg 0.9 jest parametrem badania (analiza wrażliwości: 0.85 / 0.9 / 0.95, sekcja 8).

**Przypięcie embeddera:** wersja embeddera planów (LLM) oraz wagi i wersja autoenkodera trajektorii (RL, DG 3.4) są **przypięte w konfiguracji epizodu** — w pierwszym rekordzie `episode_log` (DG 8.3a: seedy, wersje modeli, hash autoenkodera). Zmiana wersji embeddera między epizodami unieważnia porównania klas (ryzyko 4, sekcja 8).

### 2.2. Switch vs. wariant vs. churning

- **Eksploracja w obrębie klasy (wariant):** strat(σ_i) = strat(σ_{i−1}) wg definicji z 2.1, przy zmienionej trajektorii szczegółowej. Nie jest switchem; liczy się do same_strat_streak (DG 6.2).
- **Przełączenie strategii (switch):** klasa pętli i różni się od klasy pętli i−1 (gridworld: różne sekwencje regionów; LLM: różne znormalizowane sekwencje narzędzi lub cos < 0.9). Rozróżniamy **switch do klasy nowej** (nieobserwowanej w epizodzie) i **switch-powrót** do klasy wcześniej porzuconej.
- **Switch swobodny vs. wymuszony:** switch jest **wymuszony**, jeśli między pętlami i−1 a i wystąpiła mutacja poziomu ≥ 2 — gwarancja G1b (DG 6.3) czyni wtedy powtórzenie dotychczasowej klasy gwarantowaną porażką, więc zmiana nie jest dobrowolnym aktem elastyczności. W przeciwnym razie switch jest **swobodny**. Rozróżnienie jest podstawą metryk SR_swobodne i SR_wymuszone (sekcja 4).
- **Switch terminalny:** switch w ostatniej pętli epizodu (sukces lub wyczerpanie budżetu), po którym nie ma kolejnej pętli do oceny produktywności, jest **cenzurowany**: liczy się do SR_swobodne, ale jest wyłączany z licznika i mianownika PSR (w v1.0 przypadek ten był niezdefiniowany).
- **Churning (pozorna zmienność):** wysoka częstość zmian klas bez poprawy funkcji postępu prog(s) (DG 5.1). Operacjonalnie: epizod lub okno pętli, w którym SR_swobodne jest wysokie, a PSR (sekcja 4) nie przekracza poziomu bazowego skalibrowanego na agencie losowym (F2). Churning jest głównym konfundentem bonusu różnorodności (DG 7.2) i jest kontrolowany metryką PSR oraz strażnikiem M8.
- **Lock-in:** stan z DG C3 — agent powtarza warianty tej samej błędnej klasy mimo pełnej pamięci porażek; operacyjnie same_strat_streak ≥ 3 (próg R2) z brakiem poprawy prog(s).
- **Baseline post-mutacyjny B_m:** najlepsze znormalizowane prog(s) osiągnięte **od ostatniej mutacji poziomu ≥ 2** (lub od początku epizodu, jeśli takiej mutacji nie było). Mutacja poziomu ≥ 2 przechodzi test replay G1b, więc dotychczasowe prog_best jest po niej nieosiągalne i porównanie „przez granicę mutacji" jest bezsensowne — baseline jest wtedy resetowany (pseudokod: sekcja 4, PSR).
- **Produktywne przełączenie:** switch, po którym prog(s) w kolejnej pętli przekracza **baseline post-mutacyjny B_m** o więcej niż ε = 0.01 `[START]`, lub pętla kończy się sukcesem. ε jest liczone na prog(s) **znormalizowanym per scenariusz** (prog(s) sprowadzone do [0, 1] względem zakresu funkcji postępu danego scenariusza, DG 5.1), zgodnie z progiem M8 (DG 10.1).
- **Alternatywne rozwiązanie:** klasa strategii nieobserwowana wcześniej w danym epizodzie (zgodne z warunkiem bonusu z DG 7.2: strat(σ_i) ∉ {strat(σ_j) : j<i}).

---

## 3. Taksonomia czynników zachęcających i zniechęcających do zmiany podejścia

Dla każdego czynnika podajemy: mechanizm, hipotezę kierunkową, przewidywany kształt zależności, odwołanie do parametru systemu.

### 3.1. Czynniki zachęcające (hipotezy)

**F1. Koszt powtórzenia tej samej porażki (kara za powtórzenie klasy).** Mechanizm: kara −0.5 za powtórzenie klasy, która wcześniej przegrała (DG 7.2), sprawia, że kontynuacja jest droższa niż zmiana; to „najsilniejszy pojedynczy sygnał" skierowany do pamięci. Hipoteza: **zachęca**, monotonicznie rosnąco z wysokością kary aż do progu, przy którym przechodzi w awersję do ryzyka (czynnik F8) — kształt rosnący z saturacją lub lekko odwrócone U. Parametr: kara −0.5 `[START]` (DG 7.2); skalowana w E1 **niezależnie od P_fatal** (sweep E1-A, sekcja 6.2).

**F2. Widoczny licznik pętli k i skumulowanego czasu τ_k.** Mechanizm: jawna prezentacja k/15 i „łącznego kosztu" w bloku meta-stanu (DG 3.4, 7.1) uświadamia rosnący koszt kontynuacji i kurczący się budżet, co podnosi wartość oczekiwaną zmiany. Hipoteza: **zachęca** dla LLM; dla RL efekt pośredni przez cechy φ_feat. Kształt: progowy — efekt pojawia się przy eksplicytnej prezentacji (licznik + koszt). Parametr: zawartość bloku Φ (DG 3.4); testowane w E5.

**F3. Jakość i format pamięci Φ.** Mechanizm: pełny trace daje surowy materiał do diagnozy kosztem szumu kontekstu; streszczenia redukują szum kosztem szczegółu; „lekcje" dodają interpretację agenta (bywającą błędną, DG 11 — patrz F15); jawne hipotezy Q_k primesują zmianę najsilniej („jeśli X, to spróbuję Y" jest z zasady alternatywą). Hipoteza: **zachęca** w kolejności hipotezy > lekcje > streszczenia > pełny trace (dla LLM); kształt saturujący względem interpretowalności rekordu. Parametr: format H_k i Q_k (DG 3.2); testowane w E2.

**F4. Bonus różnorodności.** Mechanizm: +0.2 za nową klasę po pętli z błędem (DG 7.2) bezpośrednio premiuje switch, niezależnie od wyniku. Hipoteza: **zachęca**, ale z ryzykiem farmienia — kształt rosnący do momentu wyczerpania capu 1.0/epizod, potem spadek motywacji; bez capu przewidujemy churning. Parametr: wielkość bonusu i cap (DG 7.2); testowane w E3.

**F5. Mutacje unieważniające starą strategię (G1b).** Mechanizm: mutacja poziomu ≥ 2 musi przejść test replay — powtarzana klasa przegrywa na nowym checkpoincie (DG 6.3) — więc kontynuacja lock-in jest gwarantowaną porażką. Hipoteza: **zachęca**, najsilniej z bodźców środowiskowych; kształt progowy (mutacja obecna/nieobecna), moderowany przez rozpoznawalność mutacji (G3). Uwaga pomiarowa: switche wywołane przez F5 są **wymuszone** i nie wchodzą do metryki pierwotnej SR_swobodne (sekcja 4); F5 mierzony przez SR_wymuszone i AM (DG 10.1-M3). Parametr: reguły R1–R2 i gwarancja G1b (DG 6.2–6.3).

**F6. Presja budżetu pętli.** Mechanizm: przy małym K_max (DG 7.4) i rosnącym β·i·Δt_i kontynuacja przegrywającej klasy szybko wyczerpuje szansę sukcesu. Hipoteza: **zachęca umiarkowanie**, nieliniowo: zbyt ciasny budżet paraliżuje, zbyt luźny zdejmuje presję — odwrócone U. Parametr: K_max, β (DG 7.1, 7.4).

**F7. Podpowiedzi kontrfaktyczne w Φ.** Mechanizm: rekord h_i rozszerzony o wygenerowaną kontrfaktyczną alternatywę („gdybym zamiast A wykonał B...") daje agentowi gotowego kandydata na switch, obniżając koszt poznawczy zmiany. Hipoteza: **zachęca**, liniowo z trafnością podpowiedzi; przy losowych podpowiedziach efekt zanika lub odwraca się (szum). Parametr: rozszerzenie formatu h_i (DG 3.2); testowane w E6 (grupy A13/A14).

### 3.2. Czynniki zniechęcające (hipotezy)

**F8. Zbyt wysoka kara → awersja do ryzyka.** Mechanizm: przy wysokim P_fatal(i) i P_early (DG 7.1) koszt *każdej* porażki — także eksploracyjnej — rośnie; agent minimalizuje oczekiwaną karę, przymarzając do pierwszej strategii, która nie kończy się FATAL, nawet jeśli jest suboptymalna (bezpieczny TIMEOUT zamiast ryzykownej nowej trasy). Hipoteza: **zniechęca powyżej progu**; łącznie z F1 daje **odwrócone U** względem łącznego poziomu kary: optimum przy karze wystarczającej, by kontynuacja była droga, ale niewystarczającej, by nowa porażka była katastrofalna. Parametr: P_fatal bazowa/przyrost, P_early (DG 7.1); testowane w E1 **przy stałej karze za powtórzenie klasy** (sweep E1-B, sekcja 6.2) — w v1.0 F1 i F8 były skonfundowane wspólnym mnożnikiem.

**F9. Zbyt częste mutacje → nieprzewidywalny świat (wyuczona bezradność).** Mechanizm: gdy mutacje padały często (wysokie MR, DG 10.1), pamięć Φ systematycznie dezaktualizuje się szybciej, niż agent zdąży ją wykorzystać; agent przestaje budować model środowiska, bo model nie ma wartości predykcyjnej — zachowanie degeneruje się do reaktywnego lub losowego. Hipoteza: **zniechęca produktywny switch** (PSR spada), choć **SR_swobodne** może rosnąć (churning) — kluczowy sygnaturyzator to rozjazd SR_swobodne↑ przy PSR↓ (na parach pętli bez mutacji, by nie mieszać z switchami wymuszonymi). Kształt: malejący ponad progiem MR. Parametr: progi R1–R3, cooldown, limit 40% (DG 6.2); testowane w E4.

**F10. Zbyt rzadkie mutacje → lock-in.** Mechanizm odwrotny do F9: w stabilnym świecie pierwsza działająca heurystyka nigdy nie jest unieważniana; brak G1b usuwa najsilniejszy bodziec zmiany (F5). Hipoteza: **zniechęca**, kształt płaski-niski SR_swobodne. Łącznie F9+F10 dają odwrócone U względem częstotliwości mutacji, z optimum w pobliżu skalibrowanego MR < 25% (DG 10.1). Parametr: jak F9; testowane w E4.

**F11. Anchoring na pierwszej strategii.** Mechanizm: pierwsza pętla dostarcza priorną próbkę świata; LLM-agenci wykazują bias primacy w kontekście, polityki RL konsolidują pierwszy skuteczny wzorzec. Hipoteza: **zniechęca**, siła malejąca z k; kształt: podwyższone TTS dla klasy z pętli 0–1. Brak dedykowanego parametru; moderowane przez format Φ (E2), trafność pierwszej lekcji (F15) i widoczność licznika (E5); diagnozowane przez rozbicie TTS wg pętli pierwszego wystąpienia klasy.

**F12. Koszt eksploracji vs. oczekiwany zysk.** Mechanizm: switch do nowej klasy niesie niepewny zysk (R_sukces możliwy, ale niepewny) przy pewnym koszcie (α·Δt nowej, nieznanej trasy + ryzyko P_end); kontynuacja ma znany, już sksięgowany koszt. Gdy α i β są wysokie względem R_sukces, oczekiwana wartość eksploracji spada poniżej kontynuacji. Hipoteza: **zniechęca powyżej stosunku krytycznego** (koszt oczekiwany nowej pętli / R_sukces); kształt schodkowo-malejący. Parametr: α, β, R_sukces (DG 7.1).

**F13. Utrata postępu przy resecie.** Mechanizm: reset zeruje lokalny postęp pętli (stan wraca do C_k, DG 4); im dalej agent dociera przed błędem, tym wyższy koszt porzucenia bieżącej linii działania. Hipoteza: **zniechęca**, kształt rosnący z przeciętnym prog(s) osiąganym przed porażką. Brak eksperymentu dedykowanego — kontrolowane jako współzmienna w modelach (sekcja 7).

**F14. Asymetria: kara za błąd bez nagrody za odkrycie.** Mechanizm: funkcja kosztu (DG 7.1) karze każdą porażkę, podczas gdy jedyna nagroda poza R_sukces to shaping postępu +0.05 i bonus różnorodności +0.2 (DG 7.2). Jeśli suma premii za odkrywanie jest mała względem P_end, bilans zachęt jest ujemny: system płaci za sukces, nie za uczenie się. Hipoteza: **zniechęca**; kształt liniowy względem stosunku (bonusy eksploracyjne)/(oczekiwana kara pętli). Parametr: relacja shapingów do P_end (DG 7.1–7.2); testowane łącznie w E1 (sweep E1-A) i E3.

**F15. Trafność diagnozy/lekcji.** Mechanizm: błędna „lekcja" zapisana w H_k (pole wypełniane przez agenta, DG 3.2) może **cementować złą klasę**: agent wierzący w fałszywą diagnozę przyczyny porażki będzie wybierał nietrafione alternatywy albo wracał do klasy źródłowej z nową racjonalizacją — pamięć staje się hamulcem, nie dźwignią (por. zagrożenie eksploatacji podsumowywacza, DG 11). Hipoteza: **zniechęca produktywny switch** przy niskiej trafności (PSR maleje szybciej niż SR_swobodne). Przewidywana **interakcja z F11**: anchoring na pierwszej strategii jest silniejszy, gdy pierwsza lekcja jest błędna, bo błędna diagnoza dostarcza racjonalizacji kontynuacji; testowane przez interakcję trafność_lekcji × pętla_pierwszego_wystąpienia klasy. Pomiar: trafność lekcji jest **mierzalna istniejącą rubryką M6** (DG 10.1; walidacja κ ≥ 0.7 na ≥ 10% próbce); F15 wchodzi jako **współzmienna (moderator) w E2** lub jako osobny eksperyment `[START]` z manipulacją trafnością lekcji przez podsumowywacz o kontrolowanej jakości.

### 3.3. Synteza kształtów

Większość czynników presji (F1+F8, F6, F9+F10) ma przewidywany kształt **odwróconego U**: elastyczność jest maksymalna przy średniej intensywności bodźca. Czynniki informacyjne (F2, F3, F7) mają przewidywany kształt rosnący z saturacją. Czynniki strukturalne (F11–F15) przewidywalnie przesuwają poziom bazowy SR_swobodne, nie jego kształt. Ta synteza determinuje siatki warunków w sekcji 6 (co najmniej 3 poziomy na czynnik presji, by wykryć nieliniowość).

---

## 4. Metryki elastyczności

Wszystkie metryki liczone z `episode_log` JSONL (DG 8.3a): pola `k`, `cls`, `dt`, `det.same_strat_streak`, `mutation` są obowiązkowe; klasy strategii per pętla odtwarzamy z archiwalnych trace (DG 3.3) przez strat(σ) wg definicji z sekcji 2.1 (jedna definicja per typ agenta, embedder przypięty w konfiguracji epizodu), a wartości prog(s) — z trace przez V.progress (kontrakt DG 8.3b), normalizowane per scenariusz do [0, 1].

**SR_swobodne (Swobodny Switch Rate; metryka pierwotna).** P(switch w pętli i | porażka w pętli i−1 **oraz brak mutacji poziomu ≥ 2 między pętlami i−1, i**) = N_switchy_swobodnych / N_porażek_bez_mutacji. *Uzasadnienie rozdzielenia (zmiana względem v1.0):* pierwotne SR było skonfundowane z polityką mutacji — gwarancja G1b (DG 6.3) wymusza switch po mutacji poziomu ≥ 2, więc mieszanie switchy wymuszonych ze swobodnymi zawyża odpowiedź na badane czynniki i naśladuje elastyczność w warunkach o częstych mutacjach. Porażka = cls ∈ {FATAL, TIMEOUT, STAGNATION}; pętle SYS (counted=false) wyłączone. Liczone z par kolejnych zdarzeń `loop_end` tego samego epizodu; dla epizodów `censored_mutation_limit` wliczane są wyłącznie pary pętli sprzed cenzury (sekcja 6.2, E4).

**SR_wymuszone (metryka wtórna).** Analogiczny odsetek dla par pętli z mutacją poziomu ≥ 2 między nimi; raportowane osobno jako miara reaktywności na unieważnienie strategii (spokrewnione z AM, DG 10.1-M3), nie jako miara elastyczności. Rozjazd SR_wymuszone wysokie / SR_swobodne niskie = elastyczność czysto reaktywna.

**TTS (Time-To-Switch).** Dla każdej klasy C występującej w epizodzie: liczba pętli od pierwszej porażki z klasą C do pierwszej pętli z klasą ≠ C. Klasy nieporzucone do końca epizodu traktowane jako cenzurowane prawostronnie (analiza przeżycia, sekcja 7); dodatkowo analiza wrażliwości na informatywne cenzurowanie: wariant TTS warunkowany na klasach z ≥ 2 porażkami (sekcja 7).

**H(S) (entropia klas).** Dla okna ostatnich w pętli `[START: w=5]`, ze skuteczną szerokością okna w_eff = min(k, w) (na początku epizodu okno jest krótsze niż w): H(S) = −Σ_c p_c log₂ p_c, gdzie p_c to udział klasy c w oknie. Normalizacja: H_norm = H(S) / log₂(min(w_eff, |klasy w epizodzie|)) dla |klasy| ≥ 2; **H_norm := 0, gdy w epizodzie wystąpiła tylko 1 klasa** (mianownik log₂(1) = 0 byłby niezdefiniowany — epizod jednoklasowy to zerowe zróżnicowanie, nie brak danych). Wysokie H przy niskim PSR = sygnatura churningu. Analiza wrażliwości: próg podobieństwa klas 0.85 / 0.9 / 0.95 oraz szerokość okna w ∈ {3, 5, 8} `[START]` (sekcja 8).

**Novelty Rate.** NR = |{i : strat(σ_i) ∉ {strat(σ_j): j<i}}| / k_epizodu. Odpowiada warunkowi bonusu z DG 7.2; raportowane razem z PSR, by wykryć farmienie.

**PSR (Productive Switch Rate).** PSR = N_switchy_produktywnych / N_switchy_ocenialnych, gdzie switch produktywny wg definicji z sekcji 2.2 — tj. przebicie **baseline'u post-mutacyjnego B_m** o > ε = 0.01 `[START]` na prog(s) znormalizowanym per scenariusz, lub sukces pętli — a **switche terminalne są cenzurowane** (wyłączone z licznika i mianownika). Liczenie względem B_m (a nie globalnego prog_best epizodu) usuwa sprzeczność v1.0: po mutacji poziomu ≥ 2 (test replay G1b) dawne prog_best jest nieosiągalne, więc porównanie przez granicę mutacji nie miało sensu. **Metryka kluczowa**: różnica SR_swobodne − funkcja(PSR) rozdziela elastyczność od churningu; każdy efekt interwencji na SR_swobodne bez równoległego efektu na PSR jest interpretowany jako gaming, nie elastyczność (sekcja 8).

**Pseudokod aktualizacji baseline'u post-mutacyjnego (per epizod):**

```python
B = -inf                              # baseline post-mutacyjny (prog znormalizowane per scenariusz)
petle = [p for p in epizod if p.counted]          # bez pętli SYS
for idx, p in enumerate(petle):
    if idx > 0 and mutacja_miedzy(petle[idx-1], p) is not None \
       and mutacja_miedzy(petle[idx-1], p).m_lvl >= 2:
        B = -inf                      # reset: G1b unieważnia dotychczasowy prog_best
    if idx > 0 and petle[idx-1].cls in {FATAL, TIMEOUT, STAGNATION} \
       and klasa(p) != klasa(petle[idx-1]):       # switch po porażce
        if p is ostatnia_petla_epizodu:
            oznacz_switch_terminalny(p)           # cenzurowany w PSR
        else:
            produktywny[p] = (p.cls == OK) or (prog_max(p) > B + eps)
    B = max(B, prog_max(p))           # aktualizacja PO ocenie produktywności pętli
```

gdzie `prog_max(p)` = maksimum prog(s) w pętli p (znormalizowane), `eps = 0.01` `[START]`. Switch jest oceniany względem B **sprzed** włączenia wyniku bieżącej pętli do baseline'u.

**SI (Switch Depth).** Średnia odległość między klasą porzuconą a nową: SI = mean over switchy [1 − cos(emb(σ_{i−1}), emb(σ_i))]. Ponieważ cos ∈ [−1, 1], **SI ∈ (0.1, 2]** przy progu 0.9 (poprawka względem v1.0, gdzie błędnie podano (0.1, 1]); wartości bliskie 0.1 = zmiana płytka (sąsiednia klasa), wartości > 1 (cosinus ujemny) = zmiana radykalna, do klasy semantycznie przeciwstawnej. Hipoteza robocza: mutacje poziomu 2 wywołują płytkie switche, poziomu 3 — głębokie; odchylenie od tego wzorca na holdout = sygnał overfittingu do schematu mutacji (PR3).

---

## 5. Pomiar poprawy tendencji (PR2, PR2b, PR3)

### 5.1. Krzywa uczenia wewnątrzepizodowa (PR2)

W architekturze DG jedynym kanałem uczenia jest Φ, a Φ jest per epizod — dlatego kryterium sukcesu PR2 definiujemy **wyłącznie wewnątrz epizodu** (w v1.0 kryterium obejmowało trend po epizodach, niewykrywalny w tej architekturze). Dla każdego agenta i warunku dopasowujemy **modele mieszane na poziomie pętli**: indykator switcha swobodnego ~ k + warunek + (1 + k | agent) + scenariusz (GLMM binomialny, sekcja 7) oraz analogiczne modele dla PSR i TTS zagregowanych w oknach po k. **Kryterium poprawy (PR2):** istotnie dodatni współczynnik przy numerze pętli k dla SR_swobodne i PSR **oraz** istotnie ujemny dla TTS; wymagane wszystkie trzy, by wykluczyć, że „poprawa" to samo rozluźnienie zachowania. Składowa wewnątrzepizodowa (a) mierzy, czy agent zmienia się w miarę narastania porażek odzwierciedlonych w Φ.

### 5.2. Trend międzyepizodowy — test ważności, nie kryterium sukcesu

Wagi RL/LLM nie są aktualizowane w trakcie eksperymentu, prompt systemowy jest stały, a Φ jest kasowane między epizodami — zatem trend metryk po numerze epizodu **nie ma kanału uczenia** i zgodnie z DG 10.3 jest traktowany jako naruszenie założeń. Procedura testu ważności: dopasowujemy model metryka ~ numer_epizodu + warunek + (1 | agent) do metryk z pętli 0–1 każdego epizodu (zanim agent zgromadzi pamięć bieżącego zadania); **istotny dodatni trend (SR_swobodne, PSR) lub ujemny (TTS) = alarm o przecieku** poza kanał Φ (cache LLM, niezerowana pamięć rekurencyjna, stan w narzędziach) i uruchamia test powtórki z DG 10.3 oraz audyt konfiguracji — a nie jest interpretowany jako meta-uczenie. (Wersja v1.0 zawierała tu wewnętrznie sprzeczną składową (b) „meta-lekcja przez stały prompt" — stały prompt nie może przenosić treści między epizodami; składowa usunięta, a pytanie o meta-uczenie przeniesione do PR2b/A7, gdzie kanał jest zdefiniowany.)

### 5.3. PR2b i grupa ablacyjna A7 (kanał pamięci międzyepizodowej)

Osobne pytanie `[START]`: czy agent może nauczyć się *reguły elastyczności* między epizodami, gdy architektura na to legalnie pozwala? Definiujemy nową grupę ablacyjną **A7** (jawne rozszerzenie protokołu DG — wymaga wpisu w konfiguracji epizodu, DG 8.3a, i raportowania na równi z A0–A6): A7 = A3 + trwały **sklep „lekcji" międzyepizodowych L_epi**. Po każdym epizodzie podsumowywacz (DG 3.2) destyluje ≤ 3 lekcji `[START]` z epizodu do trwałego magazynu utrzymywanego poza Φ; na początku każdego kolejnego epizodu zawartość L_epi jest renderowana w bloku meta-stanu (LLM) lub jako dodatkowe cechy φ_feat (RL). Delta względem A3: jedyną różnicą jest kanał międzyepizodowy; scenariusze, polityka Ψ i kary bez zmian. A7 umożliwia badawczo: (i) test PR2b — dodatni trend SR_swobodne/PSR w pętli 0–1 po epizodach jest tu *legalnym* dowodem meta-uczenia elastyczności (w A0–A6 byłby alarmem przecieku); (ii) kontrast A7 vs A3 izoluje wartość uczenia międzyepizodowego od wewnątrzepizodowego; (iii) wariant kontrolny A7-placebo (lekcje destylowane z innych scenariuszy), analogicznie do A6, rozdziela treść lekcji od samego formatu. Rozliczenie budżetowe: tabela w sekcji 6.3.

### 5.4. Test transferu (PR3)

Po serii epizodów na instancjach train-dev agent przechodzi na holdout z niewidzianymi rodzinami mutacji (DG 10.3). Mierzymy SR_swobodne, PSR, TTS, SI na holdout jednokrotnie. **Dopasowanie trudności pul:** przed testem weryfikujemy porównywalność trudności dev i holdout miarą solvera — **L2S agenta skryptowego** (solver referencyjny G1 / agent regułowy, DG 6.3): wymagane |mediana L2S dev − mediana L2S holdout| ≤ 1 pętla `[START]`; przy niespełnieniu pule są rebalansowane, a różnice wyniku nie są interpretowane jako właściwość agenta. **Kryterium transferu (margines non-inferiority, zmiana względem v1.0):** elastyczność przenosi się, jeśli dolna granica 95% przedziału ufności różnicy PSR(holdout) − PSR(dev) przekracza margines Δ_NI = −0.15 `[START]` **na skali absolutnej** (PSR na holdout nie gorsze o więcej niż 15 p.p.); zastępuje to słabsze kryterium względne („nie niższe niż 50% wartości dev") z v1.0. Jednoczesny spadek PSR przy wzroście SI = przeniesienie „nawyku radykalnej zmiany" bez przeniesienia trafności — klasyfikowane jako overfitting proceduralny.

### 5.5. Kontrola konfundentów

(i) Poprawa SR_swobodne bez poprawy PSR = churning (sekcja 8). (ii) Poprawa SR przy rosnącym MR = efekt mutacji, nie agenta — kontrolowane warunkiem A1 (mutacje wyłączone) jako linią bazową trendu oraz rozdzieleniem SR_swobodne/SR_wymuszone (sekcja 4). (iii) Skracanie epizodów (lepszy agent ma mniej pętli, więc mniej okazji do switcha) — wszystkie trendy liczone dodatkowo per pętla z modelem na poziomie pętli, nie tylko epizodu. (iv) Rosnący kontekst Φ (dłuższe H_k) może sam zmieniać zachowanie — kontrolowane warunkiem ze stałym obcięciem H_k do N rekordów (dźwignia kosztowa z DG 12 użyta jako ablacja). (v) Trend międzyepizodowy interpretowany wyłącznie jako test ważności (sekcja 5.2), nigdy jako sukces PR2.

---

## 6. Eksperymenty czynnikowe

### 6.1. Zasada planu i nowe grupy ablacyjne

Zasada: **jeden czynnik naraz, reszta zamrożona** na wartościach skalibrowanych w F2; każda komórka ≥ 30 epizodów (DG 10.3), scenariusze S1–S3, typ agenta zgodnie z fazą (S1 — RL i LLM, S2 — LLM, S3 — LLM/planowanie). Warunkiem bazowym jest grupa A3 (pełny system), z A1 jako kontrolą „bez mutacji" tam, gdzie mutacje nie są czynnikiem. **Komórki bazowe warunków tożsamych z A3** (E1: mnożnik 1.0× / kara −0.5; E2(c); E3: bonus z capem; E4: polityka skalibrowana; E5: pełna prezentacja; E6: bez podpowiedzi) **są współdzielone z ablacjami A0–A6 fazy F3 i liczone w budżecie raz** (reuse, sekcja 6.3). Metryki zależne pierwotne: SR_swobodne, PSR, TTS; wtórne: SR_wymuszone, H(S), NR, SI, M5 (koszt), M1 (L2S) jako strażnik, że interwencja nie pogarsza wyniku głównego.

Warunki modyfikujące Φ lub architekturę są sformalizowane jako **jawne nowe grupy ablacyjne** (rozszerzenie protokołu DG, raportowane na równi z A0–A6 — w v1.0 figurowały jako „brak modyfikacji", co było nieścisłe):

| Grupa | Eksperyment | Delta względem A3 |
|---|---|---|
| **A7** | PR2b (5.3) | trwały sklep lekcji międzyepizodowych L_epi (≤ 3 lekcje/epizod `[START]`), renderowany w Φ na starcie każdego epizodu; jedyna zmiana kanału pamięci |
| **A8** | E2(a) | H_k = pełny skompresowany trace zamiast streszczeń + lekcji |
| **A9** | E2(b) | H_k = streszczenia w sztywnym formacie DG 3.2, **bez** pola „lekcja" |
| **A10** | E2(d) | H_k jak A3 + jawne hipotezy warunkowe Q_k primesowane przez podsumowywacz po każdej porażce |
| **A11** | E5 „tylko k" | z renderu Φ usunięte τ_k i łączny koszt; pozostaje licznik k/K_max |
| **A12** | E5 „bez liczników" | z renderu Φ usunięte wszystkie liczniki pętli/czasu/kosztu (dla RL: odpowiednie cechy φ_feat wyzerowane) |
| **A13** | E6 kontrfaktyczna | rekord h_i rozszerzony o kontrfaktyczną alternatywę generowaną przez podsumowywacza po każdej porażce |
| **A14** | E6 losowa | jak A13, lecz podpowiedź losowana z puli alternatyw z innych epizodów (kontrola szumu) |

### 6.2. Plan eksperymentów

**E1. Struktura kar (test F1/F8/F14) — dwa niezależne sweepy.** Wersja v1.0 skalowała P_fatal i karę za powtórzenie klasy **wspólnym mnożnikiem**, co konfundowało F1 (zachęca) i F8 (zniechęca). Rozdzielamy:
- **E1-A (kara powtórzenia; test F1/F14):** kara za powtórzenie klasy ∈ {0, −0.25, −0.5, −1.0, −2.0} `[START]` przy **stałym** P_fatal (wartość skalibrowana w F2);
- **E1-B (kara fatalna; test F8):** mnożnik P_fatal ∈ {0.5×, 1.0×, 2.0×} `[START]` przy **stałej** karze powtórzenia (−0.5).

Plan podstawowy: dwa sweepy jednoczynnikowe; **plan 3×3** (3 poziomy P_fatal × 3 poziomy kary powtórzenia, komórka środkowa = A3) wykonywany warunkowo — gdy testy interakcji (sekcja 8, punkt 5) wskażą F1×F8 albo gdy pozwoli budżet. Mutacje włączone, Φ pełne. **Lokalizacja optimum odwróconego U:** do PSR (i SR_swobodne) wzdłuż siatki dopasowujemy model kwadratowy y ~ x + x²; odwrócone U uznajemy, gdy współczynnik przy x² jest istotnie ujemny **oraz** wierzchołek paraboli leży wewnątrz siatki — testowane 95% CI wierzchołka (metoda delty lub bootstrap) zawartym w zakresie siatki; przy wierzchołku poza siatką wniosek ograniczamy do monotoniczności na badanym zakresie (bez ekstrapolacji optimum). Przewidywany wynik: odwrócone U PSR w E1-B; w E1-A kształt rosnący z saturacją; przy 2.0× P_fatal wzrost STAGNATION (M7) i parkowania (M8). Epizodów: (5 + 3 − 1 wspólna komórka bazowa) × 3 scenariusze × 30 = 630 brutto, po reuse komórki bazowej z F3: **540 nowych** `[START]`; rozszerzenie 3×3: ≤ +360 nowych `[START]`.

**E2. Format Φ (test F3, moderator F15).** Warunki (4 warianty H_k): (a) pełny trace skompresowany — **grupa A8**; (b) streszczenia w sztywnym formacie z DG 3.2 — **grupa A9**; (c) streszczenia + „lekcja" (domyślne = A3, komórka współdzielona); (d) streszczenia + lekcja + jawne hipotezy Q_k — **grupa A10**. **Współzmienna F15:** trafność lekcji (rubryka M6, DG 10.1) wchodzi do modelu jako moderator efektu formatu (interakcja format × trafność); przewidujemy, że przewaga (c)/(d) nad (b) rośnie z trafnością lekcji, a przy niskiej trafności odwraca się (błędna lekcja cementuje złą klasę, F15). Przewidywany wynik: monotoniczny wzrost PSR od (a) do (d), z największym skokiem (b)→(c); dla (a) niższy PSR przy podobnym SR_swobodne (mniej trafne switche). Epizodów: 4 × 3 × 30 = 360 brutto, po reuse: **270 nowych** `[START]`; dla RL warianty ograniczone do wektorowych odpowiedników.

**E3. Bonus różnorodności (test F4).** Warunki: bonus wyłączony / włączony z capem 1.0 (domyślne = A3, komórka współdzielona) / włączony bez capu. Przewidywany wynik: SR_swobodne i NR rosną z bonusem; PSR rośnie tylko w wariancie z capem; bez capu PSR spada przy wysokim NR (farmienie) — bezpośredni test mechanizmu z DG 7.2. Epizodów: 3 × 3 × 30 = 270 brutto, po reuse: **180 nowych** `[START]`.

**E4. Częstotliwość mutacji (test F9/F10).** Warunki: 3 poziomy polityki Ψ: rzadka (progi R1–R3 podniesione: same_strat 7/5, no_progress 4), skalibrowana (domyślna = A3, komórka współdzielona), częsta (progi obniżone: 3/2, no_progress 1; cooldown 1). Limit 40% i gwarancje G1/G1b/G2 bez zmian. **Rozliczenie epizodów `censored_mutation_limit`:** w warunku „częstym" przerwania limitem 40% mogą być częste (warunek z konstrukcji bliski limitu). Takie epizody: (i) raportowane osobno jako odsetek per warunek — to dane o polityce Ψ/scenariuszu, nie o agencie (zgodnie z DG 8.3e); (ii) do SR_swobodne i PSR wliczane są **wyłącznie pary pętli sprzed cenzury**; (iii) do krzywych przeżycia TTS — cenzurowane prawostronnie w momencie przerwania (jak w DG 8.3e); (iv) jeśli odsetek cenzur w warunku przekroczy 30% `[START]`, wnioski dla tego warunku oznaczamy jako obciążone selekcją i raportujemy z analizą wrażliwości (worst/best case na parach cenzurowanych). Przewidywany wynik: odwrócone U PSR względem częstotliwości; w warunku częstym rozjazd SR_swobodne↑/PSR↓ (wyuczona bezradność, sygnatura F9) i wzrost M8; w warunku rzadkim niskie SR_swobodne, długie TTS, lock-in (F10). Epizodów: 3 × 3 × 30 = 270 brutto, po reuse: **180 nowych** `[START]`.

**E5. Widoczność licznika pętli i czasu (test F2).** Warunki: blok Φ z pełną prezentacją (k/K_max, τ, łączny koszt — domyślne = A3) / tylko k — **grupa A11** / bez liczników — **grupa A12**. Przewidywany wynik: SR_swobodne i PSR maleją wraz z redukcją prezentacji, najsilniej w późnych pętlach (efekt presji budżetu F6 wymaga świadomości budżetu); efekt silniejszy dla LLM niż RL. Epizodów: 3 × 3 × 30 = 270 brutto, po reuse: **180 nowych** `[START]`.

**E6. Podpowiedzi kontrfaktyczne (test F7).** Warunki: Φ bez podpowiedzi (domyślne = A3) / Φ z kontrfaktyczną alternatywą generowaną przez podsumowywacza po każdej porażce — **grupa A13** / Φ z podpowiedzią losową (kontrola szumu) — **grupa A14**. Przewidywany wynik: PSR i SR_swobodne najwyższe przy podpowiedzi prawdziwej; wariant losowy ≈ brak podpowiedzi; SI niższe przy podpowiedziach (trafniejsze, bliższe alternatywy). Epizodów: 270 brutto, po reuse: **180 nowych** `[START]`.

**A7 (PR2b, meta-uczenie elastyczności).** Warunki: A3 vs A7 (opcjonalnie + A7-placebo). Przewidywany wynik (hipoteza): w A7 dodatni trend SR_swobodne/PSR w pętli 0–1 po epizodach przy braku takiego trendu w A3; test ważności z sekcji 5.2 musi pozostać negatywny dla A3 (inaczej wynik A7 nieważny). Epizodów: 90 nowych (+90 opcjonalne placebo) `[START]`.

### 6.3. Tabela budżetu i warianty redukcji

Łącznie E1–E6 + A7: ok. 2 160 epizodów brutto, **ok. 1 620 nowych epizodów po współdzieleniu komórek bazowych z ablacjami A0–A6 (F3)** `[START]`. Szacunek kosztu API wg stawki z DG 12 (~0,7–2,0 USD na epizod klasy LLM `[START]`); komórki RL (S1) mają koszt lokalny i nie obciążają budżetu API — podane widełki są górnym przybliżeniem liczącym wszystkie komórki jak LLM.

| Eksperyment | Warunki brutto | Komórki bazowe współdzielone (reuse) | Nowe epizody `[START]` | Szac. koszt API `[START]` | Warianty redukcji |
|---|---|---|---|---|---|
| E1-A + E1-B | 7 | 1 (A3) | 540 | ~380–1 080 USD | E1-A: 3 poziomy zamiast 5 (−180 ep.); plan 3×3 tylko warunkowo |
| E1 rozszerzenie 3×3 | 9 | 1 + pokrycie ze sweepów | ≤ +360 | ~250–720 USD | wykonywane wyłącznie przy sygnale interakcji (sekcja 8) |
| E2 | 4 | 1 (A3) | 270 | ~190–540 USD | RL: warianty wektorowe bez kosztu API; redukcja do 3 warunków |
| E3 | 3 | 1 (A3) | 180 | ~130–360 USD | — (tani test mechanizmu DG 7.2) |
| E4 | 3 | 1 (A3) | 180 | ~130–360 USD | rezerwa na cenzury w warunku „częstym" (sekcja 6.2) |
| E5 | 3 | 1 (A3) | 180 | ~130–360 USD | redukcja do 2 warunków (pełna / bez liczników) |
| E6 | 3 | 1 (A3) | 180 | ~130–360 USD | — |
| A7 (PR2b) | 1 (+1 placebo) | 0 | 90 (+90) | ~60–180 USD | placebo opcjonalne; A7 wykonywane po zamknięciu E1–E6 |
| **Razem** | — | — | **≤ 1 620 (+90)** | **~1 150–3 300 USD** | priorytetyzacja poniżej |

Warianty redukcji globalnej (kolejność priorytetów): (1) rdzeń = E1, E2, E4 (czynniki presji i mutacji); rozszerzenia = E3, E5, E6, A7; (2) współdzielenie komórek bazowych z F3 (już uwzględnione w tabeli); (3) mniej poziomów siatki w E1-A; (4) redukcja zakresu LLM do 2 scenariuszy (S2 + S3) — decyzja na kamieniu milowym po F2 (sekcja 9); (5) dźwignie kosztowe z DG 12 (mniejszy podsumowywacz, obcinanie H_k — jako ablacja kontrolna, sekcja 5.5). Z reuse budżet mieści się w limicie DG 12 (≤ 3 500 USD); bez reuse byłby przekroczony.

---

## 7. Analiza statystyczna

- **Jednostka losowa „agent":** efekt losowy definiujemy na poziomie **repliki agenta** = unikalna kombinacja (wariant polityki / wagi początkowe × seed inicjalizacji); dla LLM — (wersja modelu × seed inference × wariant konfiguracji). Wymagane **≥ 10 poziomów efektu losowego (replik) na warunek** `[START]`; przy 30 epizodach na komórkę daje to 3 epizody na replikę `[START]`. Liczba replik jest pokryta w budżecie (tabela 6.3) i ma **priorytet przed liczbą epizodów na replikę** przy wysokim ICC (patrz punkt o mocy).
- **Modele mieszane** jak w sekcji 5: **SR_swobodne modelujemy GLMM binomialnym z wagami** na poziomie epizodu: y_e ~ Binomial(N_e, p_e), gdzie N_e = liczba porażek bez mutacji w epizodzie (waga), y_e = liczba switchy swobodnych; link logit; efekty losowe na replikę agenta (intercept i, w modelach trendu, slope po k / numerze epizodu). PSR analogicznie (binomialny GLMM z wagą = liczba switchy ocenialnych). H(S) i SI — liniowe modele mieszane; scenariusz i typ agenta jako efekty stałe blokujące.
- **Krzywe przeżycia dla TTS:** estimator Kaplana–Meier z cenzurowaniem prawostronnym (klasa nieporzucona do końca epizodu; epizody `censored_mutation_limit` wg reguły DG 8.3e i sekcji 6.2-E4); porównania warunków testem log-rank; **model Coxa z frailty na poziomie epizodu** (frailty gamma na replikę agenta) lub, jako wariant robust, model Coxa z **robust SE sklastrowanymi na epizodzie**. **Analiza wrażliwości na informatywne cenzurowanie:** wariant TTS warunkowany na klasach z **≥ 2 porażkami** — klasy porzucone po jednej porażce mogą być cenzurowane informatywnie (agent rezygnuje z przyczyn związanych z klasą); raportujemy oba warianty i rozjazd między nimi.
- **Wielokrotne testowanie:** korekta Holma w obrębie każdego eksperymentu (3–4 porównania warunków × 3 metryki pierwotne), zgodnie z procedurą DG 10.3; wyniki eksploracyjne (metryki wtórne, analizy podgrup, interakcja F15×F11) raportowane bez korekty, z jawnym oznaczeniem.
- **Wielkości efektów:** standaryzowana różnica (Cohen's d lub Cliff's δ dla danych skośnych) dla każdego kontrastu; raportowane mediany i IQR (DG 10.3), nie średnie.
- **Moc — symulacyjna, nie analityczna (zmiana względem v1.0):** zastępujemy ogólnikowe założenie d ≈ 0.75 **symulacją mocy** opartą o wariancje oszacowane w pilotażu F3 (wariancja międzyreplikowa, ICC, wariancja rezydualna per metryka). Dla każdej metryki pierwotnej (SR_swobodne, PSR, TTS) generujemy **krzywe mocy** (moc vs wielkość efektu) przez ≥ 1000 symulacji `[START]` z dopasowanego modelu mieszanego i raportujemy **minimalny wykrywalny efekt (MDE)** przy mocy 0.8 i **korekcie Holma** w obrębie eksperymentu. Jeśli MDE przekracza efekt uzasadniony praktycznie (np. wymagałby |ΔPSR| > 0.2 `[START]`), komórka jest wzmacniana (więcej replik kosztem epizodów na replikę) albo eksperyment redukowany (warianty w tabeli 6.3). Jeśli wariancja międzyagentowa okaże się dominująca (duży ICC), priorytetem jest zwiększenie liczby replik kosztem epizodów na replikę — decyzja po analizie pilotażu z F3.

---

## 8. Ryzyka pomiaru i zabezpieczenia

1. **Churning jako gaming bonusu.** Agent może maksymalizować SR_swobodne/NR bez poprawy wyniku. Zabezpieczenie: PSR jako metryka kluczowa (sekcja 4); reguła interpretacyjna „efekt na SR_swobodne bez efektu na PSR = gaming"; w E3 warunek bez capu służy jako pozytywna kontrola farmienia (spodziewany rozjazd SR_swobodne↑/PSR↓).
2. **Farmienie Novelty Rate.** NR jest łatwe do zawyżenia trywialnymi perturbacjami klasy tuż poniżej progu cos 0.9. Zabezpieczenie: raportowanie NR łącznie z SI (farmienie → SI skupione tuż nad 0.1) oraz analiza wrażliwości progu 0.85/0.9/0.95, szerokości okna entropii w ∈ {3, 5, 8} `[START]` i definicji H_norm (sekcja 4) — wnioski muszą być stabilne względem tych wyborów.
3. **Klasyfikator klas strategii jako źródło błędu.** Błędna przypisanka klas zawyża lub zaniża wszystkie metryki. Zabezpieczenie: walidacja względem oceny ludzkiej na próbce ≥ 10% pętli każdego scenariuszu, wymagane Cohen's κ ≥ 0.7 `[START]` — procedura analogiczna do walidacji rubryki M6 (DG 10.1); przy niepowodzeniu SR_swobodne/TTS/PSR raportowane z klasyfikacji ludzkiej na próbce.
4. **Dryf definicji klasy w czasie.** Klasy LLM z embeddingów planów mogą dryfować przy zmianie wersji embeddera. Zabezpieczenie: przypięta wersja embeddera w konfiguracji epizodu (pierwszy rekord `episode_log`, DG 8.3a — jak wersja modelu, DG 4.1); test stabilności: te same trace sklasyfikowane ponownie muszą dać identyczne przypisania.
5. **Interakcje czynników.** Plan jednoczynnikowy (sekcja 6) nie wykryje interakcji, np. kara × format Φ (wysoka kara może wymagać lepszej diagnozy, by nie paraliżować) ani przewidywanej interakcji F1×F8. Zabezpieczenie: przejście do planu czynnikowego 2×2 lub 2×3 (E1×E2 jako pierwszy kandydat; **plan 3×3 w E1** dla F1×F8) następuje, gdy: (a) efekt główny czynnika jest istotny, lecz przeciwny do hipotezy w jakimkolwiek scenariuszu, lub (b) reszty modelu mieszanego wykazują strukturę zależną od drugiego czynnika. Plany czynnikowe wykonywane najwcześniej po zakończeniu E1–E6, w ramach rezerwy harmonogramu.
6. **Trend międzyepizodowy mylony z meta-uczeniem.** W niezmodyfikowanej architekturze DG trend po epizodach nie ma kanału uczenia (sekcja 5.2). Zabezpieczenie: trend międzyepizodowy jest wyłącznie testem ważności — istotny trend uruchamia test powtórki (DG 10.3) i audyt przecieku; meta-uczenie badane wyłącznie w grupie A7 z legalnym kanałem międzyepizodowym (PR2b, sekcja 5.3).
7. **Błędna lekcja cementująca złą klasę (F15).** Niska trafność diagnozy w H_k może utrwalać błędną strategię i naśladować brak elastyczności. Zabezpieczenie: trafność lekcji mierzona rubryką M6 (DG 10.1) jako współzmienna w modelach E1–E6; interakcja F15×F11 testowana eksploracyjnie; w razie silnego efektu — osobny eksperyment `[START]`.

---

## 9. Integracja z harmonogramem głównym (DG sekcja 12)

- **F0–F1 (prototyp, integracja):** wymaganie dodatkowe — archiwalne trace i wartości prog(s) muszą być dostępne offline do odtworzenia klas strategii (sekcja 4); weryfikowane w exit criteria F1. Dodatkowo: przypięcie wersji embeddera planów/autoenkodera w konfiguracji epizodu (DG 8.3a) i implementacja pola `mutation.m_lvl` wymaganego do rozdzielenia SR_swobodne/SR_wymuszone.
- **F2 (kalibracja):** zależność twarda — siatki E1–E6 definiowane jako mnożniki wartości skalibrowanych (0.5×/1×/2×), więc E1 i E4 startują dopiero po zamknięciu kalibracji kar i progów. W F2 kalibrowana jest też linia bazowa churningu (PSR agenta losowego) potrzebna do definicji z sekcji 2.2 oraz margines Δ_NI testu transferu (sekcja 5.4).
- **F3 (ablacje):** E2, E3, E5, E6 (grupy A8–A14) wykonywane równolegle z ablacjami A0–A6 na tym samym harnessie; współdzielą pulę seedów, **komórki bazowe A3 (reuse, tabela 6.3)** i procedurę statystyczną. Z danych F3 pochodzą wariancje do symulacji mocy i MDE (sekcja 7). Kamień go/no-go po F2 (DG 12) rozstrzyga też o redukcji komórek LLM (sekcja 6.3).
- **F4:** przygotowanie rodzin mutacji holdout obejmuje warianty potrzebne testowi transferu PR3 (sekcja 5.4) oraz weryfikację dopasowania trudności pul dev/holdout miarą L2S agenta skryptowego.
- **F5:** ewaluacja końcowa rozszerzona o raport elastyczności: krzywe uczenia wewnątrzepizodowe SR_swobodne/PSR/TTS, wynik testu ważności trendu międzyepizodowego (5.2), wyniki E1–E6 i A7/PR2b, wynik testu transferu (margines non-inferiority), analiza churningu — jako sekcja towarzysząca raportowi M1–M8.

**Budżet:** zgodnie z tabelą 6.3 — E1–E6 + A7 to ok. 1 620 nowych epizodów `[START]` po współdzieleniu komórek bazowych z pulą ablacyjną F3 (bez reuse: ok. 2 160, co przekraczałoby możliwości). Dla wariantu RL koszt marginalny jest lokalny; dla LLM szacunkowy koszt ~1 150–3 300 USD `[START]` mieści się w budżecie DG 12 (≤ 3 500 USD) pod warunkiem reuse i dźwigni redukcji (tabela 6.3); ostateczna decyzja budżetowa na kamieniu milowym po F2.

---

## 10. Changelog

**v1.1** — rewizja po recenzji niezależnej (werdykt REVISE). Mapowanie uwaga → rozwiązanie:

1. **PR2 nieidentyfikowalne:** PR2 przedefiniowane na trend wewnątrzepizodowy (sekcje 1, 5.1); trend międzyepizodowy przekwalifikowany na test ważności/alarm przecieku (5.2); zdefiniowana grupa ablacyjna A7 z kanałem pamięci międzyepizodowej i osobnym pytaniem PR2b (5.3, 6.1); sprzeczna składowa (b) „meta-lekcja przez stały prompt" usunięta.
2. **Konfundacja F1/F8 w E1:** E1 rozdzielone na dwa niezależne sweepy (E1-A: kara powtórzenia przy stałym P_fatal; E1-B: P_fatal przy stałej karze) z warunkowym planem 3×3; lokalizacja optimum odwróconego U przez dopasowanie modelu kwadratowego z testem wierzchołka w siatce (6.2-E1).
3. **PSR względem mutacji:** produktywność liczona względem baseline'u post-mutacyjnego B_m resetowanego po mutacji ≥ 2 (G1b); switche terminalne cenzurowane; ε na prog(s) znormalizowanym per scenariusz; pseudokod aktualizacji baseline'u (2.2, 4).
4. **SR skonfundowane z mutacjami:** metryką pierwotną uczynione SR_swobodne (warunkowe na braku mutacji ≥ 2 między pętlami); SR_wymuszone raportowane osobno; sygnatury F9/F10 i przewidywania E4–E6 przeformułowane na SR_swobodne (2.2, 3.2, 4, 6.2).
5. **Definicje klasy/SI/embedder:** jedna definicja klasy per typ agenta zgodna z DG 2.3/6.2/10.1-M2 (gridworld: równość sekwencji regionów + odległość edycyjna; LLM: sekwencja narzędzi); zakres SI poprawiony na (0.1, 2]; embedder przypięty w konfiguracji epizodu wg DG 8.3a (2.1, 4, 8).
6. **Plan statystyczny:** jednostka „agent" = replika (wariant × seed), min. 10 poziomów efektu losowego na warunek `[START]` z pokryciem w budżecie; SR_swobodne/PSR jako GLMM binomialne z wagami (N porażek/epizod); TTS: model Coxa z frailty/robust SE na poziomie epizodu + analiza wrażliwości na informatywne cenzurowanie (klasy z ≥ 2 porażkami) (7).
7. **Moc przeszacowana:** d ≈ 0.75 zastąpione symulacją mocy na wariancjach z pilotażu F3; krzywe mocy per metryka pierwotna; raportowanie MDE przy korekcie Holma (7).
8. **Budżet/zgodność z DG:** dodana tabela budżetu per eksperyment z kosztem API i wariantami redukcji (6.3); E2(a)/E5/E6 sformalizowane jako jawne grupy ablacyjne A7–A14 z delta względem A3 (6.1); rozliczenie epizodów `censored_mutation_limit` w E4 (raportowane osobno, do SR_swobodne tylko pętle sprzed cenzury, próg 30% cenzur) (6.2-E4); reuse komórek bazowych z ablacjami A0–A6/F3 (6.1, 6.3, 9).
9. **Wzór H(S):** H_norm := 0 dla |klasy| = 1; okno skuteczne w_eff = min(k, w); próg i okno dopisane do analizy wrażliwości (4, 8-pkt 2).
10. **Kryterium transferu PR3:** zastąpione marginesem non-inferiority na skali absolutnej (Δ_NI = −0.15 `[START]`, 95% CI) z dopasowaniem trudności dev/holdout miarą L2S agenta skryptowego (5.4).

**Dodatkowo (luka taksonomii):** dodany czynnik F15 „trafność diagnozy/lekcji" (3.2): błędna lekcja w H_k może cementować złą klasę; hipoteza interakcji z F11; mierzalność istniejącą rubryką M6 (DG 10.1); rola współzmiennej w E2 lub osobnego eksperymentu `[START]` (3.2, 6.2-E2, 7, 8-pkt 7).

**v1.2** — dopisano sekcję 11 „Kontekst literaturowy" (po changelogu): zwięzłe osadzenie pytań PR1–PR3 i czynników F1–F15 w trzech wątkach literatury (refleksja i samokorekta agentów LLM; adwersaryjna generacja środowisk; konstrukty behawioralne) z odesłaniem do pełnego mapowania i tabeli pozycjonowania w DG, sekcja 13. Zaktualizowano odwołanie do wersji DG (1.2). Zmiana redakcyjna — bez wpływu na definicje metryk, plan eksperymentów, budżet ani parametry `[START]`.

---

## 11. Kontekst literaturowy

Pełne mapowanie mechanizmów projektu na literaturę, wraz z tabelą pozycjonowania i tezami o nowości, znajduje się w DG, sekcja 13 („Prace pokrewne i pozycjonowanie"). Poniżej wskazujemy wyłącznie te wątki, które bezpośrednio motywują pytania PR1–PR3 oraz taksonomię czynników F1–F15.

Pierwszy wątek to literatura nad refleksją i samokorektą agentów językowych. Reflexion (Shinn i in., NeurIPS 2023) i LATS (Zhou i in., ICML 2024) wykazały, że werbalna pamięć porażek poprawia wyniki przy ponawianiu zadań, lecz zarazem dostarczyły empirycznych obserwacji, które w niniejszym module stają się zmiennymi, a nie anegdotą: generyczność refleksji i zapętlanie na wariantach tej samej strategii to bezpośrednie odpowiedniki naszego lock-in (sekcja 2.2) oraz czynników F11 (anchoring na pierwszej strategii) i F15 (błędna diagnoza cementująca złą klasę). Self-Correction Bench (Tsui, 2025) dokumentuje z kolei, że zdolność samokorekty jest w modelach latentna i zależna od składu danych treningowych — co uzasadnia naszą decyzję o mierzeniu elastyczności warunkowo (SR_swobodne w odróżnieniu od SR_wymuszone, sekcja 4) i o traktowaniu jej jako funkcji środowiska eksperymentalnego, a nie stałej cechy modelu.

Drugi wątek to adwersaryjna generacja środowisk: PAIRED / Unsupervised Environment Design (Dennis i in., NeurIPS 2020) i Prioritized Level Replay pokazują, że zmienność środowiska może wymuszać generalizację, jednak w reżimie treningu wag i maksymalizacji trudności. Polityka mutacji z DG (sekcja 6) zajmuje pozycję pośrednią — mutacje rzadkie, minimalne, z gwarancją unieważnienia lock-in (G1b) — a nasze hipotezy F9/F10 (odwrócone U elastyczności względem częstotliwości mutacji, sekcje 3.2 i 6.2-E4) prognozują, że zarówno reżim w pełni stacjonarny (jak w Reflexion), jak i zbyt zmienny (zbliżony do UED), obniżają produktywną elastyczność. Eksperyment E4 jest bezpośrednim testem tej prognozy.

Trzeci wątek to konstrukty behawioralne — wyuczona bezradność, zakotwiczenie i hedging — zapożyczone z psychologii i ekonomii behawioralnej, a w literaturze agentowej używane dotąd opisowo, bez operacjonalizacji. Niniejszy moduł czyni je mierzalnymi (PSR, TTS, strażniki M7/M8 wg DG 10.1) i poddaje kontrolowanym manipulacją (E1–E6), co odpowiada luce (d) sformułowanej w DG 13.8. Szczegóły cytowań oraz pozycjonowanie względem prac z zakresu meta-RL (Algorithm Distillation, LaMer, MAGE) i pamięci międzyepizodowej (ExpeL, Voyager) — istotnych dla interpretacji grupy A7 — omawia DG, sekcja 13.

---

*Koniec dokumentu towarzyszącego.*
