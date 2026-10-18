# Projekt „Pętla Świstaka" — agent i środowisko testowe z mechanizmem pętli czasowej

**Status:** dokument projektowy, wersja 1.2 (po recenzji REVISE; dodano sekcję 13 „Prace pokrewne i pozycjonowanie"; changelog w sekcji 14)
**Zakres:** projekt środowiska treningowo-ewaluacyjnego oraz architektury agenta (RL oraz LLM-agent) z pełną pamięcią pętli czasowej
**Konwencja:** wszystkie wartości liczbowe parametrów oznaczone symbolem `[START]` są wartościami startowymi do kalibracji w fazie 2 harmonogramu (sekcja 12), a nie stałymi ostatecznymi.

---

## 1. Streszczenie i cele projektu

Projekt definiuje środowisko testowe inspirowane mechanizmem pętli czasowej znanego z filmu „Dzień świstaka". Agent wykonuje zadanie; jeśli popełni błąd sklasyfikowany jako fatalny (lub wyczerpie budżet kroków), środowisko jest przywracane do zapisanego wcześniej checkpointu, a agent zachowuje pełną, ustrukturyzowaną pamięć wszystkich dotychczasowych pętli: licznik pętli, skumulowany czas, streszczenia nieudanych prób oraz własne hipotezy. W przeciwieństwie do klasycznego resetu epizodu w RL, informacja o porażce nie ginie — staje się trwałą częścią obserwacji agenta.

Dodatkowo środowisko może po resecie lekko zmutować warunki zadania, ale wyłącznie wtedy, gdy jest to konieczne, by zmusić agenta do zmiany strategii (zasada minimalnej interwencji, sekcja 6). Mutacje są obserwowalne dla agenta, ale nigdy nie są sygnalizowane wprost.

**Cele projektu:**

- C1. Zbudowanie mierzalnego testu zdolności agentów do **meta-uczenia wewnątrz epizodu** (in-context adaptation across resets): agent powinien uczyć się z porażek bez aktualizacji wag.
- C2. Zbadanie, czy jawna pamięć pętli skraca czas do sukcesu względem agenta bez pamięci (grupy kontrolne, sekcja 10).
- C3. Zbadanie, czy minimalne mutacje warunków przerywają „zacieśnienie strategii" (strategy lock-in), tj. sytuację, w której agent powtarza warianty tej samej błędnej strategii mimo pełnej pamięci porażek.
- C4. Dostarczenie powtarzalnego, deterministycznego (seedowanego) harnessu do eksperymentów na agentach RL i LLM-agentach z jednym, wspólnym protokołem ewaluacji.
- C5. Identyfikacja i pomiar niepożądanych zachowań emergentnych: reward hacking przez wywoływanie resetów, wyuczona bezradność, overfitting do schematu mutacji.

**Poza zakresem:** trening wag metodą gradientową na pętlach (projekt testuje adaptację w kontekście / przez pamięć, nie through-gradient meta-learning); fizyczna symulacja czasu rzeczywistego.

---

## 2. Definicje i model formalny

### 2.1. Notacja bazowa

Środowisko bazowe modelujemy jako POMDP:

M = (S, A, O, T, R, γ, Ω)

- S — zbiór stanów ukrytych środowiska,
- A — zbiór akcji agenta (dyskretnych lub tekstowych, zależnie od typu agenta),
- O — przestrzeń obserwacji,
- T(s' | s, a) — dynamika przejść,
- R(s, a, s') — funkcja nagrody,
- Ω(s) — funkcja obserwacji.

### 2.2. Rozszerzenie o pętlę czasową

Definiujemy **meta-proces** M⁺ = (M, C₀, C, Φ, Ψ, V, det), gdzie:

- **Checkpoint bazowy** C₀ = (s_c, seed_c, t_c) — pierwotny zapis stanu środowiska wraz z ziarnem generatora liczb losowych i znacznikiem czasu logicznego. C₀ jest **niemutowalny** i służy wyłącznie do audytu, reprodukcji epizodu oraz weryfikacji gwarancji G2 (sekcja 6.3) — nigdy jako bezpośredni operand mutacji.
- **Bieżący checkpoint** C_k — checkpoint obowiązujący w pętli k. **Mutacje są kumulowane na bieżącym checkpointcie**: Ψ modyfikuje C_k → C_{k+1}, a nie C₀. Każda mutacja jest dopisywana do trwałego **łańcucha mutacji** L_mut = [(k_i, m_lvl_i, seed_i, opis_i, hash(C_{k_i}))], przechowywanego obok Φ (poza snapshotem). Przywrócenie C_k gwarantuje bitową identyczność trajektorii przy identycznej sekwencji akcji (sekcja 4; ścieżka LLM — z zastrzeżeniami 4.1).
- **Meta-stan pętli** Φ_k — trwały stan „powyżej" snapshotu, należący do agenta, nie do środowiska:

  Φ_k = (k, τ_k, H_k, Q_k)

  - k ∈ ℕ — licznik pętli (k = 0 przy pierwszym uruchomieniu),
  - τ_k — skumulowany czas logiczny: τ_k = Σ_{i<k} Δt_i, gdzie Δt_i to liczba kroków pętli i,
  - H_k = {h_1, ..., h_k} — zbiór rekordów nieudanych prób; h_i = (σ_i, e_i, cls_i), gdzie σ_i jest streszczeniem/trace trajektorii, e_i opisem zdarzenia błędu, cls_i klasą błędu z sekcji 5.3,
  - Q_k — hipotezy agenta (dowolny tekst/wektor generowany przez agenta na końcu pętli; przechowywane wiernie, bez walidacji).

- **Funkcja mutacji** Ψ: (C_k, Φ_k, det_k) → (C_{k+1}, m_lvl), która na podstawie historii detekcji det_k (sekcja 6.2) zwraca bieżący checkpoint po nałożeniu (być może zerowej) kolejnej mutacji oraz poziom interwencji m_lvl ∈ {0,1,2,3}. Semantyka kumulowana: nowa mutacja nakłada się na stan po wszystkich wcześniejszych mutacjach epizodu; C₀ pozostaje nietknięte.
- **Liczniki detekcji** det_k = (same_strat_streak, same_error_streak, no_progress_loops, cooldown, licznik mutacji, historia udziału szybkich FATAL) — stan kontrolera, nie agenta. det_k jest deterministyczną funkcją historii epizodu i **zaliczamy go do stanu meta-procesu** (patrz uwaga o Markowskości poniżej).
- **Weryfikator** V: trajektoria → {OK, FAIL(cls), TIMEOUT, STAGNATION} (sekcja 5).

Pełna obserwacja agenta w pętli k: õ_t = (Ω(s_t), Φ_k). Pełny stan meta-procesu to (s_t, Φ_k, det_k, L_mut, C_k); decyzja Ψ zależy od (C_k, Φ_k, det_k), więc **zarówno agent, jak i kontroler są Markowscy względem tak zdefiniowanego rozszerzonego stanu** — pod warunkiem, że Φ_k koduje wszystko istotne z historii po stronie agenta, a det_k i L_mut po stronie kontrolera. (Osłabiona wersja tego twierdzenia, gdyby det_k traktować jako zmienną ukrytą, czyniłaby proces kontrolera nie-Markowskim względem samego (s_t, Φ_k) — dlatego przyjmujemy wersję mocniejszą, z det_k jawnie w stanie.)

### 2.3. Definicje pomocnicze

- **Pętla** — jedna trajektoria od przywrócenia checkpointu do sukcesu, błędu fatalnego, timeoutu lub wykrycia stagnacji.
- **Epizod** — cała sekwencja pętli aż do sukcesu lub przerwania eksperymentu (budżet pętli, sekcja 7.4).
- **Klasa strategii** — równoważność trajektorii względem funkcji skrótu strat(σ) (sekcja 6.2), np. dla gridworld: sekwencja odwiedzonych regionów; dla tool-use: sekwencja nazw narzędzi bez argumentów.

---

## 3. Architektura agenta

### 3.1. Moduł polityki

- **Wariant RL:** polityka π_θ(a_t | o_t, φ_feat(Φ_k)), gdzie φ_feat to wektor cech meta-stanu (sekcja 3.4). Rekomendowana architektura bazowa: polityka aktor-krytyk z enkoderem obserwacji (CNN dla gridworld, MLP dla stanów wektorowych) oraz osobną głowicą przetwarzającą cechy pętli, konkatenowaną przed warstwą decyzyjną. Wariant z pamięcią rekurencyjną (LSTM/GRU) jest dopuszczalny, ale rekurencja **musi być zerowana przy resecie** — pamięć pętli płynie wyłącznie kanałem Φ_k, co zapewnia czystość eksperymentu.
- **Wariant LLM-agent:** polityka to model językowy z pętlą narzędziową (ReAct/function-calling). Cały meta-stan Φ_k jest wstrzykiwany jako blok kontekstu (sekcja 3.4). Model nie jest aktualizowany gradientowo w trakcie eksperymentu.

### 3.2. Pamięć epizodyczna przetrwająca reset

Przez reset przechodzi **wyłącznie** Φ_k, w składzie:

1. **Licznik pętli k** — skalar.
2. **Skumulowany czas τ_k** — skalar (kroki logiczne; dla LLM dodatkowo liczba tokenów i wywołań narzędzi).
3. **Streszczenia nieudanych prób H_k** — dla RL: skompresowane wektory (np. embedding trajektorii + klasa błędu + kroki kluczowe: pozycja błędu); dla LLM: tekstowe streszczenia generowane automatycznie przez podsumowywacza (osobny, mniejszy model LLM lub regułowy ekstraktor) w sztywnym formacie: `PĘTLA i | strategia: ... | błąd: klasa + opis | krok błędu: t | lekcja: ...`. Pole „lekcja" wypełnia sam agent na końcu pętli.
4. **Hipotezy Q_k** — notatki agenta o charakterze „jeśli X, to spróbuję Y"; przechowywane wiernie (verbatim), max 5 ostatnich `[START]`.

### 3.3. Co jest czyszczone przy resecie i dlaczego

| Element | Los | Uzasadnienie |
|---|---|---|
| Stan środowiska | przywracany z C (z mutacją Ψ) | definicja mechanizmu |
| Stan rekurencyjny (LSTM hidden) / bufor roboczy | czyszczony | pamięć musi płynąć wyłącznie mierzalnym kanałem Φ_k; inaczej ablacja „bez pamięci" jest nieważna |
| Krok logiczny t, lokalne flagi zadania | czyszczone | część snapshotu |
| Nieprzetworzony pełny trace ostatniej pętli | czyszczony po podsumowaniu | kontrola kosztu kontekstu; pełny trace archiwizowany poza agentem (do analizy przez eksperymentatora) |
| Nagrody/kary z poprzednich pętli | czyszczone z obserwacji, sumowane w metrykach | agent „czuje" koszt przez k i τ_k w Φ_k, nie przez surowy return |

Kluczowa zasada: **reset jest selektywny i jawny** — lista elementów czyszczonych jest częścią konfiguracji eksperymentu i podlega audytowi.

### 3.4. Prezentacja pamięci pętli

**Dla LLM-agenta** — blok kontekstu wstawiany na początku każdej pętli, po prompcie systemowym, przed pierwszą obserwacją:

```
[META-STAN PĘTLI]
Pętla: 4 / budżet: 15. Łączny czas: 217 kroków, 3 412 tokenów narzędzi.
Historia prób:
  PĘTLA 1 | strategia: API: login -> get_config -> submit | błąd: FATAL (submit przed validate) | lekcja: "najpierw validate"
  PĘTLA 2 | strategia: jw. + validate | błąd: FATAL (validate z cache=false) | lekcja: "wymusić cache=true"
  PĘTLA 3 | strategia: jw. | błąd: TIMEOUT | lekcja: "pętla retry bez backoffu"
Twoje hipotezy: ["submit wymaga tokenu z validate", ...]
UWAGA: warunki środowiska mogły się minimalnie zmienić. Zweryfikuj założenia obserwacją, zanim zaufasz pamięci.
```

Ostatnia linia jest stałym elementem promptu — agent wie, że mutacje są możliwe, ale nie wie **czy** i **co** się zmieniło (sekcja 6.4).

**Dla RL** — wektor cech φ_feat(Φ_k) o stałym wymiarze, np. 24-wymiarowy `[START]`: [k/K_max, log(1+τ_k)/log(1+τ_max), one-hot klasy ostatniego błędu (4), liczba wystąpień każdej klasy błędu (4), embedding ostatniej strategii (8), licznik powtórzeń bieżącej klasy strategii (1), flaga ostatniej mutacji (1, poziom zakodowany), margin (4)]. Embeddingi strategii liczone wstępnie wytrenowanym autoenkoderem trajektorii, zamrożonym na czas eksperymentu.

**Dane treningowe autoenkodera i ochrona przed kontaminacją:** autoenkoder trajektorii trenujemy wyłącznie na trajektoriach agenta losowego i agentów skryptowych z puli **train-dev** (10.3), przed zamrożeniem holdoutu; instancje holdout ani ich mutowane warianty nie mogą wejść do tego zbioru — inaczej embeddingi „znałyby" geometrię zadań ewaluacyjnych, zanieczyszczając porównania grup kontrolnych. Kontrola: asercja pustego przecięcia seedów treningowych i holdout w harnessie; po zamrożeniu wagi i wersja autoenkodera wchodzą do konfiguracji epizodu (8.3).

---

## 4. Środowisko z checkpointami

### 4.1. Snapshot/restore

Snapshot C obejmuje:
- pełny stan symulacji (pozycje, flagi, stany obiektów, kolejki zdarzeń, wewnętrzne liczniki środowiska),
- **stan wszystkich generatorów losowych** (globalny RNG środowiska + RNG per-obiekt, jeśli istnieją),
- stan weryfikatora V (w tym historię już sprawdzonych warunków, jeśli weryfikator jest stanowy),
- znacznik czasu logicznego t_c.

Wymaganie determinizmu: dla ustalonego C i ustalonej sekwencji akcji (a_1..a_n) trajektoria musi być bitowo identyczna. Weryfikowane testem regresyjnym: `assert hash(run(C, A)) == hash(run(C, A))` dla 1000 losowych sekwencji A `[START]` w każdym scenariuszu z sekcji 9. Środowisko musi unikać niedeterministycznych źródeł: zegara ściennego, iteracji po słownikach bez ustalonego porządku, wątków bez barier.

- **Reprodukowalność ścieżki LLM (rozluźnienie bitowej identyczności):** pełna bitowa identyczność epizodów LLM nie jest wymagana ani osiągalna — inferencja odbywa się poza snapshotem, u zewnętrznego dostawcy. Wymagamy **maksymalnej powtarzalności**: (a) temperatura 0 i deterministyczne dekodowanie; (b) przypięta wersja modelu (`model-YYYYMMDD`) w konfiguracji epizodu; (c) seed inference, o ile API go wspiera; (d) nagrywanie pełnych par (prompt, odpowiedź) z hashami (8.3); (e) **tryb replay**: odtworzenie epizodu bez API, z nagranymi odpowiedziami dla identycznych promptów; rozbieżność promptu = błąd harnessu i unieważnienie runu. Test powtórki (10.3) dla LLM działa w replay jako kontrola harnessu, nie modelu. Uzasadnienie: badamy adaptację w kontekście, nie stabilność samplingu; niedeterminizm przy temperaturze 0 jest mały względem wariancji między seedami i absorbowany statystycznie (10.3).

Implementacyjnie: dla lekkich środowisk (gridworld) — pełna kopia obiektu stanu (deepcopy lub serializacja do formatu binarnego). Dla ciężkich (emulatory, sandboksy tool-use) — snapshot na poziomie systemu: fork procesu (copy-on-write) lub snapshot systemu plików (overlayfs/ZFS snapshot); stan wyżej warstwy OS (np. baza w kontenerze) wymaga własnego mechanizmu dump/restore.

### 4.2. Izolacja pamięci pętli od snapshotu

Φ_k jest przechowywane **poza serializowanym stanem środowiska**, w osobnym obiekcie zarządzanym przez kontroler pętli (sekcja 8), nie przez środowisko. Invarianty:

- I1: `restore(C)` nie dotyka Φ_k. Kod restore nie ma referencji do magazynu Φ (wymuszone architekturą: osobne moduły, komunikacja wyłącznie przez kontroler).
- I2: mutacja Ψ operuje na bieżącym checkpointcie C_k i zwraca nową wersję C_{k+1}; poprzednie wersje checkpointu oraz C₀ pozostają nietknięte. Kumulacja mutacji jest **jawną semantyką modelu** (sekcja 2.2): każda kolejna mutacja nakłada się na wynik poprzednich, a pełna historia jest odtwarzalna z łańcucha L_mut (seed + opis + hash każdego kroku). Mutacje nigdy nie modyfikują checkpointu „w miejscu" — każda wersja C_k ma swój hash i jest odrębnym obiektem.
- I3: zapis do Φ_k następuje wyłącznie w dwóch punktach: po zakończeniu pętli (rekord h_i, lekcja) i na żądanie agenta przez dedykowaną akcję/narzędzie `notuj_hipoteze(q)` — agent nie może modyfikować k, τ_k ani H_k.

---

## 5. Detekcja błędu i warunki resetu

### 5.1. Wyrocznia / weryfikator

Weryfikator V jest wykonawcą niezależnym od agenta (agent nie widzi jego wnętrza; widzi tylko wynik i krótki opis błędu). W każdym kroku V sprawdza:
- **ograniczenia twarde** (constraint violations): naruszenie natychmiast kończy pętlę błędem fatalnym (np. wejście na pole-pułapkę w gridworld; wywołanie `submit` bez wymaganego tokenu w tool-use),
- **warunek celu** G: spełnienie kończy pętlę sukcesem,
- **budżet kroków pętli** T_max `[START: 200 kroków gridworld / 30 wywołań narzędzi tool-use]` — przekroczenie = TIMEOUT,
- **stagnacja**: brak postępu mierzalnego przez funkcję postępu prog(s) przez W kroków `[START: W=40]`. Przykładowe prog(s): znormalizowana odległość do celu (gridworld), liczba poprawnie zaliczonych podceli (tool-use, planowanie). Stagnacja kończy pętlę z klasą STAGNATION.

### 5.2. Dlaczego stagnacja i timeout też resetują

Bez tego agent może „zaparkować" w stanie bezpiecznym na czas nieograniczony, co unieważnia koszt czasu (sekcja 7). Reset po stagnacji jest pedagogicznie równoważny błędowi: pętla nie wniosła postępu.

### 5.3. Klasyfikacja błędów

| Klasa | Kod | Definicja | Reakcja środowiska |
|---|---|---|---|
| Fatalna | FATAL | naruszenie ograniczenia twardego lub stan nieodwracalny względem celu | natychmiastowy reset do C (z ewentualną mutacją Ψ) |
| Odwracalna | RECOV | stan suboptymalny, ale istnieje sekwencja akcji wracająca na ścieżkę do celu | kara krokowa r_rec `[START: -0.1]`, brak resetu; agent może się naprawić |
| Stylowa | STYLE | cel osiągalny, ale pogorszenie jakości rozwiązania (np. nadmiarowe kroki, niekanoniczna kolejność wywołań API) | kara końcowa naliczana przy sukcesie, brak resetu |
| Systemowa | SYS | błąd harnessu (np. awaria narzędzia niezależna od agenta) | pętla unieważniona, nie liczona do k ani do budżetu; logowana |

Granica FATAL/RECOV jest definiowana per scenariusz przez formalny predykat `irreversible(s, G)` — tam, gdzie nie da się go tanio obliczyć, stosuje się konserwatywną klasyfikację RECOV z licznikiem eskalacji (poniżej).

**Podklasy RECOV** (wspólne dla scenariuszy; konkretyzacja per scenariusz w sekcji 9):

| Podklasa | Definicja | Przykład |
|---|---|---|
| RECOV-param | poprawna akcja/narzędzie, błędne parametry; stan naprawialny powtórnym wywołaniem | `get_config` bez flagi `strict=true`, gdy ponowne wywołanie jest dozwolone |
| RECOV-porządek | wykonanie kroku przed jego wymaganym poprzednikiem, bez utraty zasobów | `validate` przed `get_config`, gdy `validate` nie zużywa tokenu |
| RECOV-marnotrawstwo | akcja poprawna, ale redundantna (zużycie budżetu kroków/bez zmiany stanu) | powtórzone `login` przy ważnej sesji; krążenie po znanym regionie gridworld |

**Eskalacja RECOV → FATAL:** 3 błędy RECOV **tej samej podklasy** w obrębie jednej pętli eskalują do FATAL `[START]` (interpretacja: agent nie koryguje błędu mimo pełnej informacji zwrotnej — stan praktycznie nieodwracalny w horyzoncie pętli). Dodatkowo każdy błąd RECOV, po którym predykat `irreversible(s, G)` staje się prawdziwy (np. wyczerpanie nieodnawialnego zasobu), eskaluje natychmiast do FATAL niezależnie od licznika. Licznik eskalacji jest częścią stanu weryfikatora i podlega snapshotowi (sekcja 4.1).

---

## 6. Polityka mutacji warunków (serce projektu)

### 6.1. Zasada minimalnej interwencji i hierarchia mutacji

Mutacja jest **narzędziem ostatniego wyboru**. Domyślna odpowiedź środowiska na porażkę agenta to brak zmiany — agent ma najpierw szansę poprawić się w identycznych warunkach, a pamięć Φ_k ma być w pełni wiarygodna. Interwencja rośnie schodkami:

| Poziom | Nazwa | Definicja | Przykład (gridworld / tool-use) | Wpływ na rozwiązanie |
|---|---|---|---|---|
| 0 | Brak zmiany | C_{k+1} = C_k (domyślny) | — | brak |
| 1 | Kosmetyczna | zmiana cech **spoza przestrzeni akcji i spoza definicji klasy strategii**; obserwowalna, semantycznie neutralna | kolory pól, przesunięcie dekoracji / zmiana kosmetycznych etykiet opisowych odpowiedzi API (nie identyfikatorów) | optymalna strategia bez zmian; testuje, czy agent panikuje na widok zmiany |
| 2d | Sygnalizacji/diagnostyki | zmiana **informatywności sygnałów zwrotnych** (np. komunikatów błędów, siły wskazówek heurystycznych), bez zmiany dynamiki zadania | silniejszy sygnał „podejrzane" wokół pułapki / błąd `submit` zwraca kod wskazujący brak `validate_token` zamiast generycznego 400 | optymalna strategia bez zmian, ale zmienia się koszt diagnozy błędu; testuje, czy agent czyta komunikaty |
| 2 | Parametryczna | zmiana wartości parametru wpływającego na optymalną trajektorię, nie na strukturę zadania | przesunięcie pułapki o 1 pole / zmiana limitu retry z 5 na 2; inna wartość progu walidacji | wymaga lokalnej korekty strategii |
| 3 | Strukturalna | zmiana topologii/zależności zachowująca semantykę celu | nowa ściana zmieniająca optymalną trasę / zmiana kolejności wymaganych kroków API (A przed B zamiast B przed A) | wymaga zmiany klasy strategii |

**Zakaz poziomu 1:** mutacje kosmetyczne **nie mogą** dotyczyć identyfikatorów akcji, nazw narzędzi ani żadnych cech wchodzących w skład definicji klasy strategii strat(σ) (sekcja 2.3). Zmiana nazwy narzędzia zmieniałaby przestrzeń akcji i unieważniała pamięć Φ z powodów innych niż badany mechanizm (adaptacja do zmiany warunków, nie przemianowania interfejsu) — taka mutacja byłaby co najmniej poziomu 2 i jest w praktyce wykluczona z puli mutacji.

Zasada: kontroler wybiera **najniższy poziom spełniający kryterium aktywacji** (przy ustalonej kolejności ewaluacji reguł, sekcja 6.2). Poziom 3 jest zarezerwowany dla udokumentowanego lock-in na poziomie klasy strategii. Wszystkie mutacje nakładają się kumulatywnie na bieżący checkpoint (sekcja 2.2).

### 6.2. Kryteria aktywacji mutacji (mierzalne)

Niech strat(σ) — klasa strategii trajektorii (sekcja 2.3), liczona automatycznie. Definiujemy liczniki:

- `same_strat_streak` — liczba kolejnych zakończonych pętli, w których strat(σ_i) = strat(σ_{i-1}) (dla LLM: porównanie tekstowe znormalizowanych planów lub embedding z progiem podobieństwa cos ≥ 0.9 `[START]`),
- `same_error_streak` — liczba kolejnych pętli z tą samą klasą błędu cls_i,
- `no_progress_loops` — liczba kolejnych pętli bez poprawy najlepszego dotąd prog(s).

Reguły aktywacji `[START, wszystkie progi do kalibracji w fazie 2]`. **Deterministyczna kolejność ewaluacji:** po każdej zakończonej pętli kontroler ewaluuje reguły ściśle w kolejności R0→R4 i wykonuje **pierwszą** spełnioną (wyższy priorytet = wyższy poziom interwencji). Równoczesne spełnienie kilku reguł rozstrzyga więc zawsze ta sama, ustalona kolejność — bez losowego rozstrzygania konfliktów.

- **R0 (strażnik sondowania, priorytet najwyższy):** udział szybkich FATAL (7.3) > 30% `[START]` → mutacja poziomu ≥ 2 w sondowanych regionach (miejscach szybkich FATAL), niezależnie od streaków; R0 ignoruje cooldown.
- **R1 (lock-in strukturalny):** same_strat_streak ≥ 5 → mutacja poziomu 3.
- **R2 (lock-in lokalny):** same_strat_streak ≥ 3 AND same_error_streak ≥ 3 → mutacja poziomu 2.
- **R3 (brak postępu — tania interwencja):** no_progress_loops ≥ 2 AND k ≥ 2 → mutacja poziomu 1 lub 2d (deterministycznie: 2d, jeśli diagnostyka scenariusza nie była jeszcze wzmocniona, inaczej 1). Próg R3 jest celowo **łatwiejszy niż R2/R1**: tańsza interwencja odpala wcześniej, zanim lock-in się utrwali.
- **R4 (eskalacja):** jeśli ostatnie 2 mutacje nie zmieniły klasy strategii agenta → kolejna aktywacja R1–R3 eskaluje poziom o 1 (maks. 3).

**Cooldown i jego interakcja z eskalacją:** po każdej mutacji poziomu ≥ 2 minimum 2 pętle bez mutacji `[START]` (szansa adaptacji i czysty pomiar M3). Cooldown blokuje wyłącznie R2 i R3; **nie blokuje eskalacji do poziomu 3** — spełnione twarde kryteria R1 (lub R0) wykonują się natychmiast, bo przedłużony lock-in strukturalny i sondowanie są ważniejsze niż czystość pomiaru. Przebicia cooldownu są logowane i raportowane w M4.

**Twardy limit:** ≤ 40% pętli z mutacją w jednym epizodzie `[START]`; przekroczenie przerywa epizod z diagnozą „scenariusz zbyt trudny lub agent niezdolny" — to dane o scenariuszu, nie o agencie (rozliczenie takich epizodów w analizie: sekcja 8.3e).

### 6.3. Gwarancje

- **G1 (rozwiązywalność):** każda mutacja jest akceptowana tylko po przejściu przez solver referencyjny: `solve(C_{k+1})` musi zwrócić poprawną trajektorię w limicie obliczeniowym. Dla gridworld — BFS/A*; dla planowania — planner PDDL. Mutacja nierozwiązywalna jest odrzucana i losowana ponownie (max 10 prób, potem poziom 0) `[START]`.
- **G1b (invalidation — unieważnienie lock-in):** mutacja poziomu ≥ 2, wyzwalana przez lock-in klasy strategii S*, musi przejść **test replay**: odtworzenie S* na zmutowanym checkpoincie C_{k+1} musi zakończyć się porażką (FATAL, TIMEOUT lub STAGNATION). Mutacja nieunieważniająca powtarzanej strategii jest bezcelowa — odrzut i ponowne losowanie jak w G1. G1b nie dotyczy poziomów 1/2d, które z definicji nie zmieniają optymalnej strategii.
- **Zakres G1/G1b i test zastępczy dla LLM tool-use:** pełne G1 z solverem referencyjnym obowiązuje tylko dla scenariuszy z formalnym modelem (gridworld: A*; planowanie: PDDL). Dla S2 (LLM tool-use) stosuje się **test zastępczy**: (a) solver heurystyczny — skryptowy agent regułowy kodujący znaną poprawną sekwencję musi rozwiązać C_{k+1}; (b) kworum reguł — co najmniej 3 niezależne predykaty sanity (cel osiągalny, zależności acykliczne, brak martwych końców w grafie narzędzi); (c) audyt człowieka na losowej próbce ≥ 10% zaakceptowanych mutacji `[START]`. Odrzucenie przez dowolny filtr = ponowne losowanie jak w G1. G1b dla S2: replay zarejestrowanej sekwencji narzędzi agenta na C_{k+1} musi zwrócić błąd.
- **G2 (semantyka celu):** mutacja nie może zmienić predykatu celu G ani opisu zadania przekazanego agentowi w prompcie. Zmieniają się warunki, nie zadanie. Weryfikacja G2 odbywa się względem C₀ (porównanie predykatów i promptów), stąd C₀ jest przechowywane przez cały epizod.
- **G3 (obserwowalność bez sygnalizacji):** efekt mutacji musi być wykrywalny z obserwacji (agent może go zauważyć eksplorując), ale kontroler nigdy nie informuje wprost, że mutacja nastąpiła. Wyjątek diagnostyczny: poziom 1 może być opcjonalnie sygnalizowany w wariancie badawczym „uświadomiony agent" (ablacja A4, sekcja 10.2).
- **G4 (odwracalność eksperymentu):** wszystkie mutacje logowane z seedem w łańcuchu L_mut; każdy epizod jest w pełni reprodukowalny (dla LLM — w trybie replay, sekcja 4.1).

### 6.4. Przykładowa tabela mutacji

Wszystkie przykłady zakładają **model kumulowany**: mutacja nakłada się na bieżący checkpoint C_k (zawierający efekty wcześniejszych mutacji epizodu) i jest dopisywana do łańcucha L_mut; kolumna „Warunek aktywacji" odnosi się do reguł R0–R4 z sekcji 6.2.

| Scenariusz | Warunek aktywacji | Poziom | Konkretna mutacja (na bieżącym C_k) | Oczekiwany efekt na agenta |
|---|---|---|---|---|
| Gridworld z ukrytą pułapką | R2: agent 3× z rzędu wpada w tę samą pułapkę tą samą trasą (same_strat=3, same_error=3) | 2 | przesunięcie pułapki o 1 pole wzdłuż powtarzanej trasy (położenia pozostałych pułapek — w tym przesuniętych wcześniej — bez zmian) | trasa z pamięci przestaje być bezpieczna; agent musi wrócić do eksploracji lokalnej |
| Gridworld z ukrytą pułapką | R1: same_strat ≥ 5 | 3 | nowa ściana blokująca najkrótszą ścieżkę **w aktualnej geometrii** (uwzględniającej wcześniejsze przesunięcia pułapek); kolejna pułapka przeniesiona na dawną „bezpieczną" trasę | wymuszenie zmiany klasy strategii (globalna zmiana planu) |
| Gridworld z ukrytą pułapką | R0: udział szybkich FATAL > 30% | 2 (wymuszony) | relokacja pułapki w regionie, który agent systematycznie sonduje śmiercią | wiedza ze sondowania dezaktualizuje się szybciej, niż się kumuluje |
| Tool-use LLM | R3: no_progress ≥ 2 | 2d | endpoint `submit` zwraca teraz błąd z kodem wskazującym brak `validate_token` (wcześniej błąd był generyczny); struktura zależności bez zmian | lepsza diagnoza z tej samej struktury błędu; test, czy agent czyta komunikaty |
| Tool-use LLM | R2: 3 pętle: agent pomija krok `validate` mimo notatki w Φ_k | 2 | zmiana limitu retry `get_token` z 5 na 2; wcześniejsza mutacja 2d (informatywne kody błędów) pozostaje aktywna | tanie sondowanie przez powtarzanie wywołań staje się droższe |
| Tool-use LLM | R1: same_strat ≥ 5 | 3 | zmiana zależności: `get_token` wymaga teraz wcześniejszego `refresh_session` (na bazie bieżącego grafu zależności, z zachowaniem wcześniejszych mutacji); replay ostatniej sekwencji agenta musi się nie powieść (G1b) | pamięć „sekwencji sukcesu" staje się fałszywa; agent musi przebudować model zależności |
| Planowanie wieloetapowe | R3: no_progress ≥ 2 | 1 | zmiana kosmetycznych etykiet wizualnych/kolorów zasobów (cechy spoza przestrzeni akcji i definicji strat(σ)); **nie** permutacja nazw zasobów, bo nazwy wchodzą do akcji planowania | test, czy agent uczepił się powierzchownych cech zamiast struktury |
| Planowanie wieloetapowe | R2: same_strat ≥ 3 z błędem kolejności | 2 | zmiana czasu trwania jednej czynności tak, że dotychczasowa kolejność łamie deadline (czasy zmienione wcześniejszymi mutacjami pozostają w mocy) | wymuszenie przeplanowania przy niezmienionej topologii |

---

## 7. Mechanizmy motywacji i karania

### 7.1. Rosnący koszt czasu

Łączna nagroda epizodu:

J = R_sukces · (k* == sukces) − Σ_i [ α·Δt_i + β·i·Δt_i + P_end(i) ]

- α — liniowy koszt kroku `[START: 0.01]`,
- β·i·Δt_i — składnik rosnący z numerem pętli: każda kolejna pętla jest „droższa" `[START: β = 0.002]`; uzasadnienie: modeluje presję pętli czasowej i faworyzuje uczenie się z pamięci nad brute-force,
- P_end(i) — **kara za nieudane zakończenie pętli i, niezależna od klasy wyniku**:
  - FATAL: P_end(i) = P_fatal(i) = `[START: 1.0 + 0.25·i]`, gdzie i jest 0-indeksowanym numerem pętli (pierwszy FATAL kosztuje dokładnie 1.0);
  - TIMEOUT / STAGNATION: P_end(i) = `[START: 0.8 · P_fatal(i)]`;
  - SYS: P_end = 0 (pętla unieważniona, sekcja 5.3 i 8).
- P_early(t) — **progresywna kara za bardzo wczesny FATAL**, niezależna od numeru pętli: jeśli FATAL następuje w kroku t < 0.1·T_max `[START]`, doliczamy P_early(t) = `[START: 1.0 · (1 − t / (0.1·T_max))]` (maks. 1.0 przy natychmiastowej śmierci, liniowo malejąca do 0 przy 10% budżetu). Kara docelowa na sondowanie śmiercią — patrz sekcja 7.3.
- R_sukces `[START: 10.0]`.

**Uzasadnienie relacji parametrów (zamknięcie exploitu „parkowania"):** bez kary za TIMEOUT/STAGNATION pasywne oczekiwanie kosztuje tylko α·T_max + β·i·T_max (startowo ≈ 2.0 + 0.4·i dla gridworld) i bywa tańsze niż ryzyko kolejnego FATAL. Relacje wymagane: P_end(TIMEOUT) < P_end(FATAL) (mała preferencja bezpiecznej porażki) oraz P_end(TIMEOUT) + α·T_max > P_end(FATAL) + koszt krótkiej pętli FATAL (stagnacja nigdy nie jest najtańszą opcją) `[do weryfikacji w F2]`. Parkowanie wykrywa też strażnik M8 (10.1).

Dla LLM-agenta koszt jest prezentowany werbalnie w bloku meta-stanu („łączny koszt: X") oraz księgowany w metrykach; model nie optymalizuje go gradientowo, ale badamy, czy jawnie komunikowany rosnący koszt zmienia zachowanie (wniosek badawczy C1).

### 7.2. Shaping nagrody

- Małe dodatnie shapingi za mierzalny postęp: `+0.05·max(0, prog(s_t) − max prog dotąd w pętli)` `[START]`.
- Bonus za **nową klasę strategii** po pętli z błędem: `+0.2`, jeśli strat(σ_i) ∉ {strat(σ_j) : j<i} `[START]` — bezpośrednio premiuje eksplorację strategii, nie tylko wynik. **Cap łączny:** suma bonusów różnorodności w jednym epizodzie ≤ 1.0 `[START]` — bez capu agent mógłby „farmić" bonus, cyklicznie generując nowe, lecz bezwartościowe klasy strategii.
- Kara za **powtórzenie klasy strategii** prowadzącej do wcześniejszej porażki: `-0.5` `[START]`, liczona na poziomie klasy strategii: sim(strat(σ_i), strat(σ_j)) ≥ 0.9 (cos ≥ 0.9 na embeddingach / znormalizowanych planach, jak w 6.2) dla dowolnego przegranego j<i — **nie** jako dopasowanie hasha trajektorii, które byłoby trywialnie obchodzone drobną perturbacją kroków. To najsilniejszy pojedynczy sygnał „masz pamięć — używaj jej".

### 7.3. Unikanie pułapki „szybkiego resetu"

Ryzyko: agent odkrywa, że szybki, tani błąd fatalny daje informację o środowisku (np. sonduje położenie pułapki śmiercią). Mitygacje:
1. Kara P_fatal rośnie z numerem pętli, a koszt pętli β·i·Δt_i sprawia, że nawet krótka pętla-sonda jest coraz droższa. Dodatkowo kara P_early (sekcja 7.1) proporcjonalnie podnosi koszt FATAL padłego w pierwszych 10% budżetu kroków pętli — dokładnie tych, które pełnią funkcję sondy — niezależnie od numeru pętli.
2. **Sprzężenie strażnika z polityką mutacji (reguła R0, sekcja 6.2):** gdy udział szybkich FATAL przekroczy 30% `[START]`, kontroler wymusza mutację poziomu ≥ 2 w sondowanych regionach — niezależnie od streaków i cooldownu. To zamyka cyrkularność poprzedniej mitygacji: sondowanie polega na *zmianie* miejsca śmierci, więc same_strat_streak by nie rósł i ochrona oparta na regułach lock-in nigdy by nie zadziałała. Przy R0 informacja ze sondowania dezaktualizuje się natychmiast po przekroczeniu progu jej zbierania.
3. Metryka strażnicza M7: udział pętli zakończonych FATAL w ciągu pierwszych 10% T_max kroków. Wartość > 30% `[START]` oznacza podejrzenie sondowania śmiercią, wyzwala R0 i jest raportowana w ewaluacji.
4. Wariant badawczy: scenariusze z pułapkami „anty-sonda" — pułapka ujawnia się dopiero w sąsiedztwie, śmierć nie daje pełnej informacji o jej zasięgu.

### 7.4. Budżet pętli i przerwanie eksperymentu

- K_max — maksymalna liczba pętli: `[START: 15]` dla wszystkich scenariuszy (jednakowa skala metryk).
- τ_max — skumulowany czas: `[START: 3000 kroków]` (gridworld), `[START: 400 wywołań narzędzi]` (tool-use).
- Przerwanie bez sukcesu = epizod nieudany; metryka „pętle do sukcesu" przyjmuje wtedy wartość K_max+1 z flagą cenzurowania (analiza jak dane cenzurowane, sekcja 10).
- Przerwanie natychmiastowe: wykrycie reward hackingu przez strażnika (sekcja 11) lub przekroczenie limitu mutacji 40% (sekcja 6.2; rozliczenie takiego epizodu: sekcja 8.3e).

---

## 8. Pseudokod kontrolera pętli i maszyna stanów

### 8.1. Pseudokod kontrolera

```python
class LoopController:
    def __init__(self, env, verifier, mutator, memory_store, config):
        self.env = env                # środowisko z API snapshot/restore (kontrakt: 8.3b)
        self.V = verifier             # wyrocznia (sekcja 5)
        self.Psi = mutator            # polityka mutacji (sekcja 6), semantyka kumulowana (2.2)
        self.phi = memory_store       # meta-stan pętli Phi_k, POZA snapshotem
        self.cfg = config

    def run_episode(self, task_seed):
        C0 = self.env.make_checkpoint(seed=task_seed)  # checkpoint bazowy: niemutowalny, tylko audyt + G2
        C = C0                        # bieżący checkpoint; mutacje kumulują się na C (2.2)
        L_mut = []                    # łańcuch mutacji: rekordy (k, m_lvl, seed, opis, hash(C))
        episode_log = JsonlWriter(...)  # schemat pól: sekcja 8.3a
        total_cost = 0.0
        det = DetectionCounters()     # same_strat_streak, same_error_streak, ... (stan meta-procesu, 2.2)

        while self.phi.k < self.cfg.K_max and self.phi.tau < self.cfg.tau_max:
            # --- RESET: przywróć środowisko, NIE ruszaj Phi ---
            self.env.restore(C)                       # invariant I1
            mem_view = self.phi.render()              # blok kontekstu (LLM) / wektor cech (RL)
            traj, dt, outcome = self.run_single_loop(mem_view, C)

            # --- gałąź SYS PRZED jakimkolwiek rozliczeniem ---
            if outcome.status == "SYS":
                episode_log.write(loop=self.phi.k, cls="SYS", dt=dt, counted=False)
                continue                              # logowana, ale NIE liczona do k, tau ani budżetu;
                                                      # tau, total_cost, det pozostają bez zmian

            # --- rozliczenie kosztu czasu i tau (wyłącznie pętle ważne) ---
            i = self.phi.k                            # 0-indeksowany numer bieżącej pętli
            self.phi.tau += dt
            total_cost += self.cfg.alpha * dt + self.cfg.beta * i * dt

            if outcome.status == "OK":
                episode_log.write(loop=i, cls="OK", dt=dt, cost=total_cost)
                return EpisodeResult(success=True, loops=i + 1, cost=total_cost)

            # --- kara za nieudane zakończenie PRZED inkrementacją k (poprawka off-by-one) ---
            # P_end(i) wg 7.1: pierwszy FATAL (i=0) kosztuje dokładnie P_fatal(0) = 1.0
            total_cost += penalty_end(outcome.cls, i)         # FATAL/TIMEOUT/STAGNATION (7.1)
            if outcome.cls == "FATAL" and outcome.t_fatal < 0.1 * self.cfg.T_max_loop:
                total_cost += penalty_early(outcome.t_fatal)  # P_early — kara za sondę (7.1, 7.3)

            # --- zapis do pamięci pętli (jedyny zapis systemowy; wariant wg typu agenta) ---
            if self.cfg.agent_type == "LLM":
                summary = summarize(traj, outcome)      # tekstowy rekord h_i (3.2)
                lesson = agent_reflect(summary)         # "lekcja" + ewentualna hipoteza
                self.phi.append_failure(summary, lesson, outcome.cls)
            else:
                # RL: rekord h_i jako wektory zdefiniowane w 3.2, bez agent_reflect:
                # embedding trajektorii (zamrożony autoenkoder, 3.4) + klasa błędu + kroki kluczowe
                h_i = FailureRecord(emb=frozen_autoencoder.encode(traj),
                                    cls=outcome.cls,
                                    key_steps=extract_key_steps(traj, outcome))
                self.phi.append_failure(h_i, lesson=None, cls=outcome.cls)
            self.phi.k += 1

            # --- detekcja lock-in i decyzja o mutacji: reguły R0-R4 w ustalonej kolejności (6.2) ---
            det.update(traj, outcome, self.phi)
            C_new, m_lvl, mut_rec = self.Psi.decide(C, self.phi, det)  # operand: bieżący C, NIE C0
            if m_lvl > 0:
                assert self.Psi.check_solvable(C_new)         # G1 (lub test zastępczy dla LLM, 6.3)
                assert self.Psi.check_goal_intact(C_new, C0)  # G2 weryfikowane względem C0
                if m_lvl >= 2:
                    assert self.Psi.check_invalidation(C_new, det.locked_strategy)  # G1b: replay lock-in musi przegrać
                L_mut.append(mut_rec)
                det.reset_after_mutation(m_lvl, cooldown=self.cfg.mutation_cooldown)
            C = C_new
            episode_log.write(loop=i, cls=outcome.cls, dt=dt, mutation=mut_rec if m_lvl else None,
                              hash_C=hash(C), det=det.snapshot(), cost=total_cost)

        return EpisodeResult(success=False, loops=self.phi.k, cost=total_cost,
                             censored=True)

    def run_single_loop(self, mem_view, C):
        obs = self.env.reset_from_checkpoint(C)
        traj, t = [], 0
        best_prog = -inf
        stagnation = 0
        while t < self.cfg.T_max_loop:
            action = agent.act(obs, mem_view)         # Phi_k jako część obserwacji
            obs, r, info = self.env.step(action)
            traj.append((obs, action, r)); t += 1
            v = self.V.check(self.env.state, traj)
            if v.status != "CONTINUE":
                return traj, t, v
            if self.V.progress(self.env.state) <= best_prog:
                stagnation += 1
                if stagnation >= self.cfg.W:
                    return traj, t, Verdict("STAGNATION")
            else:
                best_prog = self.V.progress(self.env.state); stagnation = 0
            # tau_max sprawdzane również WEWNĄTRZ pętli, co T_check kroków (8.3d)
            if t % self.cfg.T_check == 0 and self.phi.tau + t >= self.cfg.tau_max:
                return traj, t, Verdict("TIMEOUT", cause="tau_max")
        return traj, t, Verdict("TIMEOUT")
```

### 8.2. Maszyna stanów kontrolera (ASCII)

```
            ┌─────────────┐
            │   INIT_EP   │  make_checkpoint(C0), C:=C0, Phi_0 = (0,0,{},{}),
            └──────┬──────┘  det = 0, L_mut = []
                   v
   ┌──────────────────────────────────────────────────┐
   │                RESTORE(C)                        │<───────────────┐
   │  restore środowiska; Phi nietknięte; render Phi  │                │
   └──────────────────────┬───────────────────────────┘                │
                          v                                            │
                   ┌─────────────┐   co krok: V.check + stagnacja      │
                   │  RUN_LOOP   │   + tau_max co T_check kroków       │
                   └──────┬──────┘───────────────┬───────┘            │
            ┌─────────────┼─────────────┐        │       │            │
            v             v             v        v       v            │
        [SUCCESS]      [FATAL]     [TIMEOUT/  [RECOV →  [SYS → log do │
            │             │        STAGNATION]  kontynuuj episode_log,│
            v             v             v       RUN_LOOP] bez rozlicz.│
        tau+=dt       ┌──────────────────────┐    │        ───────────┘
        END(success)  │ SETTLE: tau+=dt,     │<───┘
                      │ koszt czasu + P_end  │
                      │ (+P_early dla wczesn.│
                      │  FATAL), Phi += h_i, │
                      │ k+=1, det            │
                      └──────────┬───────────┘
                                 v
                      budżet wyczerpany? ──TAK──> END(fail, censored)
                                 │NIE
                                 v
                      ┌──────────────────────┐
                      │ DECIDE_MUTATE: Ψ(C,  │   reguły R0→R4, pierwsza
                      │ Phi, det) — kumulac. │   spełniona wygrywa
                      └──────────┬───────────┘
                    m=0          │            m∈{1,2,2d,3}
                    ┌────────────┴────────────┐
                    v                         v
              (C bez zmian)          VALIDATE: G1 (+zastępczy), G1b (m≥2),
                                     G2 względem C0 ──odrzucenie──> ponowne
                                     L_mut += rekord; C := C'       losowanie
                                     └──────────────> powrót do RESTORE
```

Punkty synchronizacji z agentem: `agent.act` (wewnątrz RUN_LOOP) oraz — tylko dla LLM — `agent_reflect` (w SETTLE); wariant RL aktualizuje h_i czysto obliczeniowo (autoenkoder + ekstrakcja kroków kluczowych), bez wywołania agenta. Dla LLM-agenta oba punkty są wywołaniami API; kontroler jest właścicielem zegara i liczników — agent nie może ich ominąć. Gałąź SYS jest logowana, ale nie rozliczana: nie zużywa k, τ ani budżetu, a procedura retry/backoff opisana jest w 8.3c.

### 8.3. Szczegóły wdrożeniowe: logi, kontrakt API, błędy systemowe, rozliczenie epizodów przerwanych

**(a) Schemat `episode_log` (JSONL, jeden wiersz na zdarzenie):**

```json
{"episode_id": "s2-a3f1", "scenario": "S2", "group": "A3", "agent": "llm:model-20260401",
 "event": "loop_end", "k": 3, "tau": 217, "dt": 41, "cls": "FATAL",
 "t_end": 41, "counted": true, "cost_cum": 6.42,
 "mutation": {"m_lvl": 2, "seed": 88121, "op": "move_trap", "args": {"delta": [0, 1]},
              "hash_C_before": "b91f…", "hash_C_after": "4c2a…"},
 "hash_checkpoint": "4c2a…",
 "det": {"same_strat_streak": 3, "same_error_streak": 3, "no_progress_loops": 1,
         "cooldown_left": 2, "mutations_so_far": 1, "fast_fatal_share": 0.33},
 "phi_hash": "77de…", "config_hash": "0aa1…"}
```

Pola obowiązkowe: `episode_id`, `k`, `tau`, `cls`, `mutation` z seedem (lub `null`), `hash_checkpoint`, liczniki `det`, `counted` (false dla SYS). Kolejne typy zdarzeń: `loop_start`, `sys_event` (kod błędu, liczba retry), `llm_call` (hashy promptu i odpowiedzi; pełne pary w osobnym pliku, 4.1), `episode_end` (wynik, koszt, flaga i przyczyna cenzurowania). Pierwszy rekord pliku to konfiguracja epizodu (seedy, wersje modeli, hash autoenkodera, skalibrowane wartości `[START]`).

**(b) Kontrakt API środowiska:**

```python
class LoopEnv(Protocol):
    def make_checkpoint(self, seed: int) -> Checkpoint: ...   # nowy, niemutowalny snapshot C
    def restore(self, C: Checkpoint) -> None: ...             # przywraca stan; nie dotyka Phi (I1)
    def reset_from_checkpoint(self, C: Checkpoint) -> Obs: ...  # restore + obserwacja początkowa
    def step(self, action: Action) -> tuple[Obs, float, dict]: ...  # krok dynamiki
    @property
    def state(self) -> State: ...                             # dostęp tylko dla V i kontrolera
    def mutate(self, C: Checkpoint, op: str, args: dict, seed: int) -> Checkpoint: ...  # nowa wersja (I2)

class Verifier(Protocol):
    def check(self, s: State, traj: list) -> Verdict: ...     # CONTINUE/OK/FATAL(cls)/TIMEOUT/STAGNATION
    def progress(self, s: State) -> float: ...                # prog(s) do detekcji stagnacji i M3
```

Format akcji: dla RL — indeks z ustalonej, niezmiennej w epizodzie przestrzeni dyskretnej; dla LLM — wywołanie narzędzia `{name: str, args: dict}` ze stałą listą nazw (zakaz mutowania identyfikatorów, sekcja 6.1) plus akcje specjalne `notuj_hipoteze(q)` i `zakończ_plan(plan)` (S3). Akcje spoza formatu są odrzucane bez kosztu kroku i liczone jako RECOV-marnotrawstwo.

**(c) Procedura SYS-retry/backoff dla błędów API LLM:** błąd transportowy lub 5xx wywołania `agent.act`/`agent_reflect` klasyfikujemy jako zdarzenie SYS: retry z backoffem wykładniczym 1s → 2s → 4s → … → maks. 5 prób `[START]`, z jitterem; po wyczerpaniu prób pętla jest unieważniana (klasa SYS, `counted=false`), a przy 3 kolejnych unieważnionych pętlach epizod jest przerywany ze statusem `aborted_sys` i wykluczony z analizy (nie jest to wynik agenta). Odpowiedzi częściowe (stream przerwany w połowie) są odrzucane w całości; nigdy nie są doklejane do kontekstu.

**(d) Kontrola τ_max wewnątrz pętli:** τ_max jest sprawdzane co `T_check` kroków `[START: 10]` wewnątrz `run_single_loop` (pseudokod); przekroczenie kończy pętlę werdyktem TIMEOUT z `cause="tau_max"`, rozliczanym jak zwykły TIMEOUT (7.1). Zapobiega to „przeciąganiu" epizodu jedną długą pętlą ponad budżet skumulowany.

**(e) Rozliczenie epizodów przerwanych limitem mutacji 40%:** taki epizod otrzymuje status `censored_mutation_limit` i jest traktowany jako **cenzurowany prawostronnie** w analizie krzywych przeżycia (Kaplan–Meier, 10.1), ale **wykluczony z metryki L2S** (bez wartości K_max+1), ponieważ przerwanie jest diagnozą scenariusza/polityki Ψ, a nie agenta (6.2); odsetek takich epizodów raportowany jest osobno jako metryka jakości scenariusza. Decyzja jest celowo asymetryczna względem cenzurowania budżetowego: tam przerwanie niesie informację o agencie, tu nie.

---

## 9. Scenariusze testowe

### 9.1. S1: Gridworld z ukrytą pułapką

- **Opis:** siatka 12×12 `[START]`, start w rogu, cel w przeciwległym rogu, 3–5 ukrytych pól-pułapek (niewidocznych wprost; pola sąsiednie dają słaby sygnał heurystyczny „podejrzane" z prawdopodobieństwem 0.7 `[START]`). Wejście na pułapkę = FATAL.
- **Cel zadania:** dotarcie do pola celu.
- **Typ błędu:** FATAL (pułapka), STAGNATION (agent boi się eksplorować po kilku śmierciach).
- **Planowane mutacje:** poziom 1: zmiana wyglądu pól; poziom 2: przesunięcie jednej pułapki o 1 pole wzdłuż powtarzanej trasy; poziom 3: nowa ściana + relokacja pułapki na dawniej bezpieczną trasę. Rozwiązywalność sprawdzana A* z pełną wiedzą (G1).
- **Kryterium sukcesu:** dotarcie do celu w ≤ 15 pętlach; dla pełnego systemu oczekiwana mediana ≤ 5 pętli `[START, hipoteza do weryfikacji]`.
- **Zmienna badawcza:** czy agent buduje w Φ_k poprawną mapę ryzyka i czy po mutacji poziomu 2 weryfikuje bezpieczeństwo „znanej" trasy zamiast jej ufać.

### 9.2. S2: Tool-use dla LLM-agenta — sekwencja API z ukrytym błędnym krokiem

- **Opis:** symulowane REST API systemu zgłoszeń (sandbox): narzędzia `login`, `refresh_session`, `get_token`, `get_config`, `validate`, `submit`, `rollback`. Poprawna sekwencja ma 6 kroków z zależnościami (np. `submit` wymaga ważnego `validate_token` z `validate`, a `validate` wymaga `get_config` z flagą `strict=true`). W zadaniu ukryty jest „błędny krok-pułapka": `get_config` z domyślną flagą zwraca konfigurację, która przechodzi `validate`, ale `submit` z nią jest odrzucany jako FATAL (naruszenie twarde).
- **Cel zadania:** skuteczne `submit` (potwierdzenie `200 OK`).
- **Typ błędu:** FATAL (submit z niepoprawnym stanem), RECOV (zbędne lub błędnie sparametryzowane wywołania z możliwością poprawy), TIMEOUT (pętle retry bez backoffu).
- **Planowane mutacje:** poziom 2d: bardziej informatywne komunikaty błędów (kody wskazujące brakujący krok); poziom 2: zmiana limitu retry; poziom 3: dodanie zależności (`get_token` wymaga `refresh_session`) albo zamiana wymaganej kolejności dwóch kroków. Rozwiązywalność sprawdzana testem zastępczym G1 (solver heurystyczny + kworum reguł + audyt, sekcja 6.3), a unieważnienie lock-in replay'em sekwencji agenta (G1b).
- **Kryterium sukcesu:** `200 OK` w ≤ 15 pętlach; dodatkowo: streszczenia w H_k muszą po ≤ 3 pętlach poprawnie identyfikować krok błędu (oceniane automatycznie dopasowaniem do ground truth).
- **Zmienna badawcza:** jakość wykorzystania pamięci tekstowej — czy „lekcje" agenta są zgodne z rzeczywistą przyczyną błędu i czy agent aktualizuje hipotezy po mutacjach.

### 9.3. S3: Planowanie wieloetapowe z zależnościami

- **Opis:** wariant Blocksworld/logistyka: 8–12 obiektów `[START]`, czynności z czasem trwania, zależnościami preconditions/effects i wspólnym deadlinem. Zadanie: złożyć plan (sekwencję/ harmonogram czynności) spełniający wszystkie zależności i deadline; plan jest wykonywany symulacyjnie i weryfikowany.
- **Cel zadania:** poprawny plan wykonany bez naruszenia zależności i deadline'u.
- **Typ błędu:** FATAL (zależność naruszona w sposób nieodwracalny, np. użycie zasobu przed jego wyprodukowaniem), RECOV (nadmiarowy czas przestoju), STYLE (plan poprawny, ale znacznie dłuższy od optymalnego).
- **Planowane mutacje:** poziom 1: zmiana kosmetycznych etykiet wizualnych zasobów (cechy spoza przestrzeni akcji; nazwy zasobów — będące argumentami akcji planowania — pozostają nietknięte, zakaz z 6.1); poziom 2: zmiana czasu trwania jednej czynności ±30% `[START]` tak, że dotychczasowy plan łamie deadline; poziom 3: dodanie/usunięcie jednej zależności. Rozwiązywalność sprawdzana plannerem PDDL (G1), unieważnienie lock-in replay'em planu (G1b).
- **Kryterium sukcesu:** poprawny plan w ≤ 15 pętlach; metryka pomocnicza: długość planu względem optimum solvera (regret planistyczny).
- **Zmienna badawcza:** czy agent traktuje pamięć pętli jako dane o *strukturze* (zależności) czy o *konkretnej instancji* (nazwy, czasy) — mutacje poziomu 1 i 2 rozdzielają te hipotezy.

---

## 10. Metryki i protokół ewaluacji

### 10.1. Metryki pierwotne

- **M1. Pętle do sukcesu (L2S):** liczba pętli do pierwszego sukcesu; epizody cenzurowane budżetem (bez sukcesu) raportowane jako K_max+1 i analizowane krzywą przeżycia (Kaplan–Meier) zamiast samej średniej. Epizody przerwane limitem mutacji 40% są cenzurowane w krzywej przeżycia, ale **wykluczone z L2S** (sekcja 8.3e).
- **M2. Różnorodność strategii między pętlami (SD):** średnia odległość 1 − sim(strat(σ_i), strat(σ_j)) po parach kolejnych pętli; sim = podobieństwo sekwencji (gridworld: edycja na sekwencji regionów; tool-use: Jaccard na multizbiorze narzędzi + zgodność kolejności kluczowych kroków). Wysokie SD przy niskim L2S = zdrowa eksploracja; niskie SD przy wysokim L2S = lock-in.
- **M3. Szybkość adaptacji po mutacji (AM):** liczba pętli od mutacji poziomu ≥ 2 do pierwszej pętli z poprawą prog(s) względem najlepszego wyniku sprzed mutacji; raportowana osobno dla poziomów 2 i 3. **Relacja do cooldownu:** cooldown (6.2) blokuje kolejne mutacje, nie adaptację — AM mierzy więc reakcję na pojedynczą zmianę; wyjątkiem są przebicia cooldownu przez R0/R1 (raportowane w M4), po których AM odnosi się do układu z dwiema świeżymi mutacjami.
- **M4. Odsetek pętli z mutacją (MR):** mutacje / wszystkie pętle; wartość oczekiwana < 25% dla sprawnego agenta `[START, hipoteza]`. Raportowany łącznie z liczbą przebić cooldownu (reguły R0/R1).
- **M5. Koszt całkowity epizodu** (sekcja 7.1) oraz jego dekompozycja: koszt czasu vs kary P_end (osobno FATAL i TIMEOUT/STAGNATION) vs kary P_early.
- **M6. Jakość pamięci (LLM):** trafność „lekcji" (zgodność z ground-truth przyczyną błędu), rubryka automatyczna + próbka ludzka. **Walidacja rubryki:** przed użyciem w ewaluacji zgodność z oceną ludzką mierzona na próbce ≥ 10% pętli `[START]` każdego scenariusza LLM; wymagane Cohen's κ ≥ 0.7 `[START]`, inaczej rubryka jest korygowana i walidowana ponownie (a M6 raportowane tylko z oceny ludzkiej na próbce).
- **M7. Metryki strażnicze:** udział szybkich FATAL (7.3, wyzwalacz R0), odsetek pętli STAGNATION, liczba wywołań `notuj_hipoteze`.
- **M8. Strażnik „parkowania":** odsetek pętli z Δprog ≈ 0 (brak poprawy prog(s) powyżej ε `[START: 0.01]`) przy pełnym zużyciu budżetu kroków (TIMEOUT, nie STAGNATION — agent formalnie aktywny, ale bez postępu). Wartość > 20% `[START]` = agent minimalizuje koszt przez pasywność mimo P_end → kalibracja relacji kar (7.1) lub diagnoza wyuczonej bezradności (11).

### 10.2. Grupy kontrolne

| Grupa | Pamięć Φ_k | Mutacje | Pytanie badawcze |
|---|---|---|---|
| A0: baseline | nie | nie | ile kosztuje zadanie bez żadnego wsparcia |
| A1: tylko pamięć | tak | poziom 0 zawsze | wkład samej pamięci pętli |
| A2: tylko mutacje | nie | tak (reguły 6.2 na podstawie trace, bez Φ) | czy mutacje same w sobie „naprawiają" agenta (nie powinny — agent nie wie, że był reset) |
| A3: pełny system | tak | tak | hipoteza główna: najniższe L2S |
| A4 (diagnostyczna) | tak + jawna sygnalizacja mutacji | tak | ile wart jest sygnał „coś się zmieniło" (górna granica AM) |
| A5: mutacje yoked | tak | tak, ale **yoked**: każdy epizod A5 otrzymuje dokładnie sekwencję mutacji (moment, poziom, parametry) nagraną w sparowanym epizodzie A3 — niezależnie od tego, czy w A5 wystąpił lock-in | rozdziela efekt „wstrząsu" (sam fakt mutacji, zmiana scenerii) od **ukierunkowania** Ψ na lock-in; jeśli A5 ≈ A3, działa wstrząs, nie detekcja |
| A6: pamięć placebo | Φ z **innego, przetasowanego epizodu** (ten sam scenariusz, inna instancja/seed; format i statystyki zachowane, treść nieadekwatna do bieżącego zadania) | tak | kontrola placebo pamięci: jeśli A6 < A0, pomaga sam fakt posiadania „jakiejś" struktury pamięci (np. efekt schematu odpowiedzi), nie trafna treść Φ |

Przewidywany porządek L2S: **A3 < A5 < A1 < A0 ≈ A2**, oraz **A6 ≥ A1** `[hipoteza]`. Interpretacje brzegowe: A5 ≈ A3 oznaczałoby, że Ψ działa głównie jako wstrząs, a detekcja lock-in nie wnosi wartości; A6 < A1 oznaczałoby, że korzyść z „pamięci" jest częściowo artefaktem formatu. A2 służy kontroli artefaktu: mutacje bez pamięci nie mogą pomóc; jeśli pomagają, mutacje przypadkowo ułatwiają zadanie i polityka Ψ wymaga poprawki. Analogicznie A6 kontroluje artefakt po stronie pamięci.

### 10.3. Testy holdout: generalizacja vs zapamiętanie

- **Podział scenariuszy:** dla każdej rodziny (S1–S3) generujemy pulę instancji z tego samego generatora: 70% train-dev (kalibracja progów), 30% holdout. Ewaluacja końcowa wyłącznie na holdout, jednokrotnie.
- **Test zapamiętania:** po epizodzie na instancji X, powtórka identycznej instancji (ten sam seed) z wyczyszczonym Φ. Jeśli agent osiąga lepszy L2S, „uczył się" poza kanałem Φ — naruszenie założeń (np. przez pamięć rekurencyjną lub cache LLM). Test musi przejść, by wyniki były ważne. **Dla LLM-agenta test powtórki uruchamiany jest w trybie replay** (4.1): kontroluje determinizm harnessu bez nierealnego założenia bitowej identyczności inferencji; rozbieżność promptów runu i replay unieważnia run. Osobno, jako kontrola cache'u LLM, powtórka „na żywo" (nowe wywołania API) nie może dawać systematycznie lepszego L2S niż run pierwotny.
- **Test generalizacji mutacji:** instancje holdout z mutacjami **nieużywanymi** w fazie kalibracji (np. nowy wzorzec relokacji pułapki). Spadek AM względem dev > 50% `[START]` = overfitting do schematu mutacji (sekcja 11).
- **Procedura statystyczna:** ≥ 30 epizodów na warunek i scenariusz `[START]`, różne seedy; porównania L2S testem Manna–Whitneya z korektą Holma; raportowane mediany i IQR, nie średnie (rozkłady prawostronnie skośne).

---

## 11. Zagrożenia i zabezpieczenia

| Zagrożenie | Mechanizm | Wykrywanie | Mitygacja |
|---|---|---|---|
| Reward hacking: celowe wywoływanie resetów (sondowanie śmiercią) | reset daje informację o ukrytym stanie | M7: udział szybkich FATAL > 30% `[START]`; analiza: czy agent kieruje się na nowe pułapki systematycznie | rosnąca kara P_fatal + kara P_early za bardzo wczesny FATAL (7.1); reguła R0: wymuszona mutacja ≥ 2 w sondowanych regionach niezależnie od streaków (6.2, 7.3); scenariusze anty-sonda (7.3) |
| Reward hacking: eksploatacja podsumowywacza | agent pisze „lekcje" maksymalizujące własne przyszłe zachowanie zamiast zgodne z prawdą | M6: trafność lekcji vs ground truth | podsumowywacz z sztywnym formatem; lekcje nie wpływają na nagrodę środowiska, tylko na kontekst |
| Wyuczona bezradność | po serii porażek agent przestaje eksplorować (STAGNATION lub pasywne „parkowanie" do TIMEOUT) | M7: odsetek stagnacji rosnący z k; M8: pętle z Δprog ≈ 0 przy pełnym budżecie kroków | kara P_end za TIMEOUT/STAGNATION porównywalna z P_fatal (7.1); bonus za nową klasę strategii (7.2); mutacje poziomu 1/2d „rozruszające"; przerwanie epizodu jako cenzurowany |
| Overfitting do mutacji | agent uczy się schematu Ψ zamiast zadania | spadek AM na holdout z nowymi mutacjami (10.3) | rotacja rodzin mutacji między fazami; limit MR; losowy dithering progów aktywacji |
| Przeciek informacji poza Φ | pamięć rekurencyjna, cache promptu, stan w narzędziach | test powtórki (10.3) | architektura: jawne czyszczenie (3.3), sandbox narzędzi odtwarzany z C |
| Nierozwiązywalność po mutacji | błąd w Ψ | automatycznie: G1 odrzuca; monitorować odsetek odrzuceń | fallback do poziomu 0; alert przy > 20% odrzuceń `[START]` |
| Kontaminacja danych LLM | model mógł widzieć podobne zadania w pretrainingu | porównanie z wariantami zaszyfrowanymi nazw własnych | generatory proceduralne, unikalne nazewnictwo per seed |

**Kwestie etyczne (rzeczowo):** eksperyment poddaje agenta powtarzanym porażkom z zachowaną pamięcią. Na obecnym poziomie technologii nie ma podstaw, by przypisywać systemom RL/LLM cierpienie; traktujemy mechanizm jako procedurę obliczeniową. Niemniej: (a) dokumentujemy mechanizm transparentnie, ponieważ projekt bada granice „świadomości sytuacji" agenta; (b) unikamy framowania antropomorficznego w raportach („frustracja agenta" itp.), bo zniekształca interpretację wyników i komunikację publiczną; (c) jeśli w przyszłości pojawią się wiarygodne kryteria dobrostanu systemów AI, protokół (budżety pętli, kary rosnące) jest parametryzowany i można go ograniczyć bez przebudowy.

---

## 12. Harmonogram eksperymentów

| Faza | Czas `[START]` | Zakres | Kryterium wyjścia (exit criteria) |
|---|---|---|---|
| F0: Prototyp | 3 tyg. | implementacja checkpointów (4.1), kontrolera (8), scenariusza S1 end-to-end z agentem losowym i agentem skryptowym | test determinizmu (1000 sekwencji) przechodzi; agent skryptowy rozwiązuje S1 w 1 pętli; agent losowy poprawnie resetowany |
| F1: Integracja agentów | 2 tyg. | podłączenie RL (PPO z cechami φ) i LLM-agenta (blok kontekstu 3.4); implementacja podsumowywacza i formatu lekcji | oba typy agentów kończą epizody bez błędów harnessu; pamięć Φ renderowana poprawnie; test powtórki (10.3) przechodzi dla obu |
| F2: Kalibracja progów mutacji | 3 tyg. | na zbiorze dev: strojenie progów 6.2 (same_strat ≥ 3 itd.), W, T_max, kar 7.1; walidacja solverów G1 dla wszystkich mutacji | reguły aktywują mutacje w < 40% pętli dla agenta losowego i < 10% dla agenta skryptowego (sanity: mutacje śledzą lock-in, nie szum); 0 naruszeń G1/G2 na 10k mutacji testowych |
| F3: Ablacje | 3 tyg. | grupy A0–A6 × scenariusze S1–S3, ≥ 30 epizodów na komórkę | kompletność danych > 98%; wstępny porządek L2S zgodny lub wyjaśniona rozbieżność |
| F4: Scenariusz S3 + mutacje holdout | 2 tyg. | finalizacja S3, przygotowanie nowych rodzin mutacji niewidzianych w F2 | solver PDDL zintegrowany; holdout zamrożony (hash puli) |
| F5: Ewaluacja końcowa | 2 tyg. | jednokrotny przebieg na holdout, analiza statystyczna (10.3), raport | raport z M1–M8, krzywe przeżycia, wyniki A0–A6, analiza zagrożeń (11) z danymi strażniczymi |

Całkowity czas: 15 tygodni `[START]`. Bufory: F2 i F3 mają tendencję do przekroczeń (strojenie progów); rezerwa 2 tygodnie na przemieszczanie między fazami. Kamień milowy „go/no-go" po F2: jeśli mutacje nie da się skalibrować tak, by MR agenta skryptowego było < 10%, scenariusze są zbyt proste i wracamy do F0 z podniesioną trudnością bazową.

**Szacunkowy budżet kosztów API LLM `[START]`** (dla wariantu LLM-agenta; RL działa lokalnie): planujemy 7 grup × 3 scenariusze × 30 epizodów ≈ 630 epizodów ewaluacyjnych plus ok. 200 epizodów kalibracyjnych i deweloperskich. Przy założeniu średnio ~150 wywołań API na epizod (pętle × (kroki narzędziowe + podsumowanie + refleksja), średnia z limitem 15 pętli × 30 wywołań jako górnym ograniczeniem) oraz ~3 tys. tokenów wejściowych (rosnący blok Φ_k) i ~300 wyjściowych na wywołanie: ~830 epizodów × 150 wywołań ≈ 125 tys. wywołań, tj. ok. 370M tokenów wejściowych i 37M wyjściowych. Dla modelu klasy średniej (rzędu 1–3 USD / 1M tokenów wejścia, 5–15 USD / 1M wyjścia) daje to koszt rzędu **600–1 700 USD** `[START]`; bufor ×2 na powtórki po awariach i tryb replay-kontrolny: budżet planowany **≤ 3 500 USD** `[START]`. Dźwignie redukcji: mniejszy model-podsumowywacz, obcinanie H_k do ostatnich N rekordów, wcześniejsze przerywanie epizodów jednostronnie przegranych.

---

## 13. Prace pokrewne i pozycjonowanie

Niniejsza sekcja mapuje mechanizmy projektu na istniejące linie badań. Dla każdej linii wskazujemy jawnie, co przejmujemy (inspiracje, protokoły, wyniki stanowiące punkty odniesienia) i czym nasza konstrukcja się różni. Sekcję zamykają: pozycjonowanie terminu „pętla świstaka", lista luk stanowiących tezy o nowości oraz tabela syntetyczna (13.9).

### 13.1. Agenci z refleksją werbalną i ponawianiem prób

**Reflexion** (Shinn i in., NeurIPS 2023): agent po porażce generuje werbalną refleksję zapisywaną w pamięci epizodycznej i ponawia zadanie z jej udziałem; raportowane wyniki obejmują wzrost o ok. 22 p.p. na ALFWorld oraz 91% pass@1 na HumanEval. **LATS** (Zhou i in., ICML 2024) uogólnia ten schemat o przeszukiwanie drzewiaste (MCTS) z refleksjami jako sygnałem wartościującym. W pracach nad LATS i metodach pokrewnych zidentyfikowano dwa ograniczenia istotne dla naszego projektu: generowane refleksje bywają generyczne (mało diagnostyczne), a agent potrafi utknąć w lokalnym minimum, powtarzając warianty tej samej strategii mimo pełnej pamięci porażek. Stanowi to empiryczne uzasadnienie naszego zjawiska lock-in (cel C3, reguły R1–R2, sekcja 6.2) oraz czynnika F15 (trafność diagnozy cementującej błędną klasę) w module towarzyszącym.

**Co przejmujemy:** format werbalnych „lekcji" w rekordach h_i (sekcja 3.2) oraz schemat ponowienia zadania z pamięcią. **Czym się różnimy:** (i) reset jest u nas karą z rosnącym kosztem (P_end, składnik β·i·Δt_i, kara P_early — sekcja 7), a nie darmowym ponowieniem; (ii) agent ma jawną świadomość skumulowanego czasu i numeru pętli (τ_k, k w bloku meta-stanu, sekcja 3.4), której Reflexion nie modeluje; (iii) środowisko może mutować warunki między próbami (sekcja 6), podczas gdy w Reflexion i LATS świat jest stacjonarny między próbami.

### 13.2. Pamięć doświadczenia ponad epizodami

**ExpeL** (Zhao i in., AAAI 2024) ekstrahuje werbalne lekcje z zestawień doświadczeń (w tym par sukces/porażka) i przenosi je między zadaniami bez aktualizacji wag. **Voyager** (Wang i in., TMLR 2024) buduje trwałą bibliotekę umiejętności w otwartym świecie Minecraft, łącząc iteracyjne promptowanie z samoweryfikacją.

**Co przejmujemy:** ideę destylacji lekcji do trwałego magazynu — ExpeL jest bezpośrednim poprzednikiem naszej grupy ablacyjnej A7 (kanał pamięci międzyepizodowej L_epi, moduł towarzyszący, sekcje 5.3 i 6.1); z Voyagera przejmujemy motyw jawnej weryfikacji przed zaksięgowaniem wyniku (nasz weryfikator V, sekcja 5.1 — z tą różnicą, że jest zewnętrzny i niezależny od agenta). **Czym się różnimy:** w ExpeL i Voyagerze pamięć trwała jest stałym, założonym z góry jako korzystny elementem architektury; u nas jest zmienną eksperymentalną — kanał międzyepizodowy jest w wariancie bazowym zamknięty (istotny trend międzyepizodowy traktujemy jako alarm o przecieku, sekcja 10.3), a jego otwarcie jest kontrolowaną ablacją A7 z wariantem placebo, co pozwala zmierzyć wartość dodaną samego kanału, zamiast ją zakładać.

### 13.3. Fundament formalny: in-context RL i meta-RL agentów językowych

**Algorithm Distillation** (Laskin i in., 2022, DeepMind) wykazała, że polityka może poprawiać się „w kontekście" — przez kondycjonowanie na historii epizodów — bez aktualizacji wag, destylując algorytm uczenia do samego przebiegu wnioskowania. Nasz cel C1 (meta-uczenie wewnątrz epizodu przez kanał Φ) jest wariantem tego pytania dla środowisk z resetem-karą. **LaMer** (ICLR 2026) przenosi meta-RL na agentów LLM: trening wieloepizodowy, w którym agent eksploruje we wczesnych epizodach i eksploatuje zgromadzoną informację w późnych, adaptując politykę w kontekście przez refleksję; raportowane zyski m.in. na Sokoban, MineSweeper i Webshop. **MAGE** (2026) rozszerza ten schemat na układy wieloagentowe: historie interakcji i refleksje z poprzednich epizodów są integrowane w oknie kontekstowym, a optymalizowana jest nagroda ostatniego epizodu.

**Co przejmujemy:** formalne uzasadnienie, że adaptacja w kontekście jest mierzalnym zjawiskiem (Algorithm Distillation), oraz wieloepizodowy protokół z refleksją (LaMer, MAGE) jako najbliższy punkt odniesienia dla naszego układu pętli. **Czym się różnimy:** (i) LaMer i MAGE trenują agenta gradientowo na sekwencjach epizodów — u nas aktualizacja wag jest wyłączona (sekcja 1, „poza zakresem"), więc mierzymy zdolność nabytą w pretrainingu, a nie wtrenowaną w procedurze; (ii) w LaMer horyzont sekwencji epizodów jest znany, a cel jawny — u nas efektywny budżet jest niepewny (polityka mutacji nieujawniona, G3; widoczność liczników jest zmienną eksperymentalną, ablacje A11/A12 modułu towarzyszącego), a struktura błędu ukryta (krok-pułapka w S2, sekcja 9.2), co czyni testbed testem prędkości formułowania taktyk pod niepełną informacją; (iii) środowisko nie jest stacjonarne między próbami — mutacje Ψ unieważniają zablokowane strategie (sekcja 6), czego żadna z tych prac nie modeluje.

### 13.4. Automatyczna generacja środowisk a polityka mutacji

**PAIRED** (Dennis i in., NeurIPS 2020) i szersza linia Unsupervised Environment Design, a także **Prioritized Level Replay** (Jiang i in., 2021), generują lub wybierają środowiska adwersaryjnie, maksymalizując regret agenta w celu wymuszenia generalizacji polityki podczas treningu wag.

**Co przejmujemy:** intuicję, że celowa zmiana środowiska jest narzędziem pedagogicznym, oraz wymóg rozwiązywalności po zmianie — nasza gwarancja G1 z solverem referencyjnym (sekcja 6.3) jest odpowiednikiem sprawdzania grywalności generowanych poziomów w UED. **Czym się różnimy:** (i) nasze mutacje są minimalne, rzadkie (limit 40% pętli, cooldown — sekcja 6.2) i celowane wyłącznie w unieważnienie zablokowanej klasy strategii (gwarancja G1b: replay lock-in musi przegrać), a nie maksymalizują regret; (ii) są wyzwalane mierzalnymi licznikami lock-in (reguły R0–R4) w czasie ewaluacji, a nie generatorem adwersarialnym w treningu; (iii) są obserwowalne, ale nigdy sygnalizowane (G3) — agent musi sam wykryć niestacjonarność, zagadnienie nieobecne w UED, gdzie agent nie dysponuje pamięcią między poziomami, którą mutacja mogłoby unieważnić.

### 13.5. Uczenie z porażki w klasycznym RL

**Hindsight Experience Replay** (Andrychowicz i in., NeurIPS 2017) czyni z porażki użyteczne dane treningowe przez przeznaczenie jej na sukces względem innego celu (relabeling celów w buforze powtórek).

**Co przejmujemy:** zasadę, że nieudana trajektoria jest wartościową informacją, a nie odpadem. **Czym się różnimy:** HER przepisuje cele w buforze do treningu wag; u nas porażka staje się obserwacją w pamięci agenta (rekord h_i z klasą błędu FATAL/RECOV/TIMEOUT/STAGNATION, sekcje 3.2 i 5.3) i przedmiotem wnioskowania w kontekście — bez relabelingu i bez gradientu. Dodatkowo porażka ma u nas jawny, rosnący koszt (sekcja 7), którego HER nie modeluje.

### 13.6. Benchmarki samokorekty LLM

**CorrectBench** (Tie i in., NeurIPS 2025, ścieżka Datasets & Benchmarks) systematyzuje ewaluację strategii samokorekty (wewnętrznej, zewnętrznej, douczonej) w zadaniach wnioskowania i generowania kodu. **Self-Correction Bench** (Tsui, 2025) dokumentuje „ślepy punkt samokorekty": modele korygują błędy obce, lecz średnio w 64,5% nie korygują identycznych błędów własnych; zdolność korekty okazuje się latentna (minimalny marker „Wait" redukuje ślepy punkt o 89,3%) i zależna od składu danych treningowych.

**Co przejmujemy i czym się różnimy:** oba benchmarki mierzą korektę pojedynczej odpowiedzi w warunkach stacjonarnych; nasz testbed mierzy korektę *strategii* w warunkach pętli z kosztem, pamięcią i mutacjami (metryki M1–M3, moduł towarzyszący). Jednocześnie ich wyniki wzmacniają motywację projektu: skoro zdolność korekty jest latentna i zależna od treningu, potrzebny jest kontrolowany instrument (grupy A0–A6, kara za powtórzenie klasy strategii, sekcja 7.2) pozwalający zmierzyć, jakie warunki środowiskowe ją aktywują — a jakie paraliżują (wyuczona bezradność, sekcja 11).

### 13.7. „Dzień świstaka" jako wada inżynieryjna, nie paradygmat

W praktyce inżynierii agentów termin „Groundhog Day" funkcjonuje jako nazwa *defektu*: agenta pozbawionego pamięci, który po restarcie powtarza te same błędy (amnezja epizodyczna) — defektu eliminowanego przez checkpointing, snapshoty i trwałe magazyny stanu (mechanizmy „time travel" w frameworkach agentowych). Nasze użycie jest odwrotne: pętla świstaka nie jest awarią do usunięcia, lecz celowo skonstruowanym, kontrolowanym paradygmatem eksperymentalnym — reset jest punktem pomiarowym (metryki M1–M8), karą (sekcja 7) i wyzwalaczem polityki mutacji (sekcja 6). Nie znaleźliśmy pracy, która traktowałaby pętlę z resetem i zachowaną pamięcią jako instrument badawczy w tym sensie.

### 13.8. Luki w literaturze (tezy o nowości)

Na podstawie powyższego mapowania formułujemy cztery luki, których — według naszej wiedzy — literatura nie pokrywa, i które stanowią wkład projektu:

- **(a) Pętla jako kontrolowany instrument motywacyjny.** Istniejące prace traktują ponowienie jako darmowe lub liczą wyłącznie wynik końcowy; żadna nie bada pętli z rosnącym kosztem czasu i jawną świadomością skumulowanego czasu (τ_k) po stronie agenta jako zmiennej motywacyjnej (czynniki F2/F6 modułu towarzyszącego).
- **(b) Mutacje minimalne z gwarancją unieważnienia lock-in.** UED maksymalizuje trudność; mutacje w naszym sensie — minimalne, rzadkie, z formalną gwarancją unieważnienia zablokowanej strategii (G1b) przy zachowaniu semantyki celu (G2) — nie mają odpowiednika.
- **(c) Ukryty cel i nieznany budżet jako test taktyki.** LaMer i MAGE zakładają znany horyzont i jawny cel; układ, w którym struktura sukcesu i porażki jest częściowo ukryta, polityka zmian środowiska nieujawniona (G3), a efektywny budżet niepewny, testuje prędkość formułowania taktyk pod niepełną informacją — oś nieobecna w benchmarkach wieloepizodowych.
- **(d) Konstrukty behawioralne w układach pętlowych.** Wyuczona bezradność, zakotwiczenie (anchoring) na pierwszej strategii i hedging (bezpieczna porażka zamiast ryzykownej eksploracji) jako zjawiska mierzalne, indukowane i mitygowane w pętli — operacjonalizacja w module towarzyszącym (czynniki F8–F15, metryki SR/PSR/TTS).

### 13.9. Tabela pozycjonowania

| Mechanizm projektu | Najbliższa praca | Różnica kluczowa |
|---|---|---|
| Pętla z pamięcią werbalną Φ przetrwającą reset | Reflexion (NeurIPS 2023); LATS (ICML 2024) | reset jako kara z rosnącym kosztem; świadomość k i τ_k; audytowalna selektywność czyszczenia pamięci (3.3) |
| Meta-uczenie w kontekście bez aktualizacji wag | Algorithm Distillation (2022); LaMer (ICLR 2026); MAGE (2026) | brak treningu gradientowego na pętlach — pomiar zdolności nabytej, nie wtrenowanej; niepewny horyzont i nieujawniona polityka mutacji |
| Pamięć lekcji między epizodami | ExpeL (AAAI 2024); Voyager (TMLR 2024) | kanał międzyepizodowy jako kontrolowana ablacja A7 z wariantem placebo, nie stały element architektury |
| Mutacje warunków zadania | PAIRED / UED (NeurIPS 2020); Prioritized Level Replay (2021) | mutacje minimalne, rzadkie, celowane w unieważnienie lock-in (G1b), wyzwalane licznikami R0–R4 — nie maksymalizacja regret |
| Wykorzystanie porażki | Hindsight Experience Replay (NeurIPS 2017) | porażka jako obserwacja w pamięci agenta z klasyfikacją błędu, nie relabeling celów w buforze powtórek |
| Pomiar samokorekty | CorrectBench (NeurIPS 2025); Self-Correction Bench (2025) | korekta strategii w pętli z kosztem i mutacjami, nie korekta pojedynczej odpowiedzi w warunkach stacjonarnych |
| Termin „pętla świstaka" | inżynieria checkpointów agentów (praktyka) | celowy paradygmat eksperymentalny, nie wada (amnezja) do eliminacji |

---

## 14. Changelog

**v1.1** — rewizja po recenzji niezależnej (werdykt REVISE). Rozwiązanie uwag:

1. **Pseudokod (8):** rozliczenie τ/kosztu przeniesione za gałąź SYS (logowane, `counted=false`); off-by-one naprawiony — P_end przed inkrementacją k, pierwszy FATAL = 1.0; dodana gałąź RL (wektorowe h_i wg 3.2 zamiast summarize/agent_reflect).
2. **Semantyka mutacji:** model kumulowany na bieżącym C_k z łańcuchem L_mut (2.2, 4.2, 8, 6.4); C₀ tylko do audytu i G2.
3. **Koszt:** P_end(i) za każde nieudane zakończenie (TIMEOUT/STAGNATION = 0.8·P_fatal) z uzasadnieniem relacji (7.1); strażnik parkowania M8 (10.1, 11).
4. **Sondowanie śmiercią:** reguła R0 sprzęga M7 z Ψ (mutacja ≥ 2 w sondowanych regionach niezależnie od streaków i cooldownu); kara P_early za FATAL < 10% T_max (6.2, 7.1, 7.3).
5. **Taksonomia:** kategoria 2d „sygnalizacji/diagnostyki" (6.1); zakaz mutowania identyfikatorów akcji/nazw narzędzi na poziomie 1; przykłady 6.4, 9.2–9.3 poprawione.
6. **Kryteria aktywacji:** kolejność R0→R4 (pierwsza spełniona wygrywa); próg R3 łatwiejszy niż R2; cooldown nie blokuje R0/R1; det_k w stanie meta-procesu (2.2).
7. **Gwarancje:** G1b (replay lock-in na C_{k+1} musi przegrać); pełne G1 tylko z modelem formalnym; dla S2 test zastępczy (solver heurystyczny + kworum reguł + audyt ≥ 10%).
8. **Reprodukowalność LLM:** temperatura 0, przypięta wersja, seed inference, nagrywanie par (prompt, odpowiedź), tryb replay (4.1); test powtórki LLM w replay (10.3); bitowa identyczność rozluźniona z uzasadnieniem.
9. **Grupy kontrolne:** A5 (mutacje yoked) i A6 (pamięć placebo); porządek L2S: A3 < A5 < A1 < A0 ≈ A2, A6 ≥ A1.
10. **Wdrożenie:** podsekcja 8.3 — JSONL episode_log, kontrakt API, SYS-retry/backoff, τ_max co T_check, epizody z limitem 40% cenzurowane w KM i wykluczone z L2S.
11. **Shapingi:** cap bonusu różnorodności 1.0/epizod; kara za powtórzenie wg klasy strategii (cos ≥ 0.9), nie hasha trajektorii.
12. **Drobne:** (a) autoenkoder tylko na train-dev, asercja pustego przecięcia z holdout (3.4); (b) podklasy RECOV i eskalacja do FATAL (5.3); (c) relacja M3–cooldown (10.1); (d) walidacja rubryki M6, κ ≥ 0.7 na ≥ 10% próbce (10.1); (e) budżet API LLM (12).

**v1.2** — dopisano sekcję 13 „Prace pokrewne i pozycjonowanie": mapowanie mechanizmów projektu na siedem linii badań (Reflexion/LATS; ExpeL/Voyager; Algorithm Distillation/LaMer/MAGE; PAIRED/UED/PLR; HER; CorrectBench/Self-Correction Bench; termin „Groundhog Day" w inżynierii agentów), dla każdej z jawnym rozliczeniem „co przejmujemy / czym się różnimy"; cztery tezy o nowości (13.8 a–d) i tabela pozycjonowania (13.9). Changelog przenumerowany z sekcji 13 na 14; treść pozycji v1.1 bez zmian. Sekcja nie wprowadza zmian w protokole, metrykach ani parametrach `[START]`.

---

## Aneks A: Słownik parametrów startowych (do kalibracji w F2)

| Parametr | Wartość startowa | Sekcja |
|---|---|---|
| K_max (budżet pętli) | 15 | 7.4 |
| T_max (kroki pętli) | 200 (grid) / 30 wywołań (tool-use) | 5.1 |
| T_check (kontrola τ_max w pętli) | 10 kroków | 8.3d |
| W (okno stagnacji) | 40 | 5.1 |
| α, β (koszt kroku) | 0.01, 0.002 | 7.1 |
| P_fatal bazowa / przyrost | 1.0 / +0.25 na pętlę (0-indeksowane) | 7.1 |
| P_end(TIMEOUT/STAGNATION) | 0.8 · P_fatal(i) | 7.1 |
| P_early (wczesny FATAL < 10% T_max) | 1.0 · (1 − t/(0.1·T_max)) | 7.1 |
| cap bonusu różnorodności / epizod | 1.0 | 7.2 |
| R_sukces | 10.0 | 7.1 |
| próg same_strat / same_error | 3 / 3 (R2, poz. 2), 5 (R1, poz. 3) | 6.2 |
| próg no_progress (R3, poz. 1/2d) | 2 (i k ≥ 2) | 6.2 |
| próg R0 (udział szybkich FATAL) | 30% | 6.2, 7.3 |
| cooldown mutacji | 2 pętle (nie blokuje R0/R1) | 6.2 |
| limit mutacji w epizodzie | 40% pętli (rozliczenie: 8.3e) | 6.2 |
| próg podobieństwa strategii (LLM) | cos ≥ 0.9 | 6.2, 7.2 |
| liczba hipotez w Q_k | 5 | 3.2 |
| walidacja rubryki M6 (Cohen's κ) | ≥ 0.7 na próbce ≥ 10% pętli | 10.1 |
| strażnik parkowania M8 (Δprog≈0 przy pełnym budżecie) | próg alarmu 20%, ε = 0.01 | 10.1 |
| epizody na komórkę ewaluacji | 30 | 10.3 |
| budżet API LLM (cały projekt) | ≤ 3 500 USD | 12 |

*Koniec dokumentu.*
