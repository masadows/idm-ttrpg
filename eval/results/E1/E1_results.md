# E1 — podsumowanie pomiaru bazowego

## Cel i zakres

Pomiar E1 ustanawia punkt odniesienia dla bazowego pipeline'u RAG przed testowaniem
ulepszeń retrievalu. Porównano dwie kolekcje:

- `rpg_structure` — chunking strukturalny,
- `rpg_fixed` — chunking fixed-size.

Każda konfiguracja została oceniona na dwóch poziomach:

- `source` — zgodność z dokładnym `source_id` z golden setu,
- `section` — zgodność na poziomie sekcji podręcznika.

Ewaluacja obejmowała retrieval oraz generowanie odpowiedzi (`--generate`).

## Konfiguracja

| Parametr | Wartość |
|---|---|
| Model LLM | `qwen3.5:4b` |
| Temperatura | `0.0` |
| Reasoning effort | `none` |
| Model embeddingów | `qwen3-embedding:0.6b` |
| `TOP_K` | `5` |
| Strategia chunkingu | `structure` / `fixed` zależnie od kolekcji |
| `CHUNK_SIZE` | `800` |
| `CHUNK_OVERLAP` | `120` |
| Structured output | `auto` |
| Vector store | Qdrant |
| Generowanie | włączone |
| `OLLAMA_CONTEXT_LENGTH` | `16384` |

## Zbiór testowy - golden set

Źródło: [`golden_rpg.jsonl`](../../golden_rpg.jsonl)

- 120 pytań,
- 104 pytania z oczekiwaną odpowiedzią w korpusie,
- 16 pytań spoza korpusu, dla których oczekiwana jest odmowa,
- podręczniki reprezentowane w goldenie:
  - `PlayerDnDBasicRules.pdf` — 47 odwołań,
  - `Mythras - 2E (v3.1).pdf` — 30,
  - `SW5e - Player's Handbook.pdf` — 21,
  - `DMBasicRulesv.pdf` — 14.

Zbiór obejmuje pytania różnej trudności o reguły D&D 5e, Mythras i SW5e, a także dotyczące kilku sytemów na raz oraz pytania na które nie ma odpowiedzi w korpusie.

## Wyniki

Szczegółowe wyniki znajdują się w katalogu [`eval/results/E1`](./).
W tym szczególnie tabele z wynikami dla każdego pytania:
- [`e1_rpg_structure_qwen3.5-4b.md`](./e1_rpg_structure_qwen3.5-4b.md),
- [`e1_rpg_structure_section_qwen3.5-4b.md`](./e1_rpg_structure_section_qwen3.5-4b.md),
- [`e1_rpg_fixed_qwen3.5-4b.md`](./e1_rpg_fixed_qwen3.5-4b.md),
- [`e1_rpg_fixed_section_qwen3.5-4b.md`](./e1_rpg_fixed_section_qwen3.5-4b.md).

### Retrieval

| Kolekcja | Poziom | recall@1 | recall@3 | recall@5 | MRR | ctx_chars |
|---|---:|---:|---:|---:|---:|---:|
| `rpg_fixed` | source | **0.27** | **0.36** | **0.43** | **0.33** | **3993** |
| `rpg_fixed` | section | **0.30** | **0.44** | **0.51** | **0.40** | **3993** |
| `rpg_structure` | source | **0.29** | **0.39** | **0.43** | **0.35** | **5166** |
| `rpg_structure` | section | **0.34** | **0.52** | **0.57** | **0.45** | **5166** |

- `recall@k`: jaka część oczekiwanych źródeł pojawiła się w top-k chunkach,
- `MRR (Mean Reciprocal Rank)`: średnia z 1/pozycja pierwszego trafnego chunka,
- `ctx_chars`: ile znaków kontekstu trafia do modelu (top-k chunków).


### Generowanie

| Kolekcja | Poziom | refusal_ok | citation_hit | inline_cited | latency | human_score |
|---|---:|---:|---:|---:|---:|---:|
| `rpg_fixed` | source | **0.86** | **0.36** | **0.77** | **1.07 s** | **0.57** |
| `rpg_fixed` | section | **0.88** | **0.46** | **0.80** | **1.07 s** | **0.64** |
| `rpg_structure` | source | **0.85** | **0.34** | **0.72** | **1.83 s** | **0.72** |
| `rpg_structure` | section | **0.85** | **0.43** | **0.71** | **1.79 s** | **0.69** |


- `refusal_ok`: model odpowiedział na pytanie z dokumentów / odmówił na pytanie spoza nich,
- `citation_hit`: wśród cytowanych chunków jest oczekiwane źródło,
- `inline_cited`: odpowiedź ma w tekście znaczniki [n] przy twierdzeniach,
- `latency_s`: czas retrieval + generacja,
- `human_score`: udział poprawnych odpowiedzi wśród wszystkich ocenionych ręcznie.

### Ocena ręczna
#### Fixed-size chunking
| pierwszy tag | ocenione | pominięte | poprawne | niepoprawne | poprawne odmowy | niepoprawne odmowy | human_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| complex | 20 | 0 | 6 | 8 | 0 | 6 | 0.30 |
| easy | 48 | 0 | 30 | 3 | 0 | 15 | 0.62 |
| medium | 33 | 0 | 15 | 11 | 0 | 7 | 0.45 |
| multi | 3 | 0 | 2 | 0 | 0 | 1 | 0.67 |
| unanswerable | 16 | 0 | 0 | 0 | 16 | 0 | 1.00 |
| **suma** | **120** | **0** | **53** | **22** | **16** | **29** | **0.57** |

#### Fixed-size chunking, ocena na poziomie sekcji
| pierwszy tag | ocenione | pominięte | poprawne | niepoprawne | poprawne odmowy | niepoprawne odmowy | human_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| complex | 20 | 0 | 6 | 7 | 0 | 7 | 0.30 |
| easy | 48 | 0 | 32 | 3 | 0 | 13 | 0.67 |
| medium | 33 | 0 | 21 | 4 | 0 | 8 | 0.64 |
| multi | 3 | 0 | 2 | 1 | 0 | 0 | 0.67 |
| unanswerable | 16 | 0 | 0 | 0 | 16 | 0 | 1.00 |
| **suma** | **120** | **0** | **61** | **15** | **16** | **28** | **0.64** |

#### Structure-based chunking
| pierwszy tag | ocenione | pominięte | poprawne | niepoprawne | poprawne odmowy | niepoprawne odmowy | human_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| complex | 20 | 0 | 9 | 3 | 0 | 8 | 0.45 |
| easy | 48 | 0 | 36 | 2 | 0 | 10 | 0.75 |
| medium | 33 | 0 | 23 | 7 | 0 | 3 | 0.70 |
| multi | 3 | 0 | 2 | 1 | 0 | 0 | 0.67 |
| unanswerable | 16 | 0 | 0 | 0 | 16 | 0 | 1.00 |
| **suma** | **120** | **0** | **70** | **13** | **16** | **21** | **0.72** |

#### Structure-based chunking, ocena na poziomie sekcji
| pierwszy tag | ocenione | pominięte | poprawne | niepoprawne | poprawne odmowy | niepoprawne odmowy | human_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| complex | 20 | 0 | 10 | 4 | 0 | 6 | 0.50 |
| easy | 48 | 0 | 33 | 5 | 0 | 10 | 0.69 |
| medium | 33 | 0 | 22 | 6 | 0 | 5 | 0.67 |
| multi | 3 | 0 | 2 | 1 | 0 | 0 | 0.67 |
| unanswerable | 16 | 0 | 0 | 0 | 16 | 0 | 1.00 |
| **suma** | **120** | **0** | **67** | **16** | **16** | **21** | **0.69** |

## Wnioski

1. **Przewaga chunkingu strukturalnego (`rpg_structure`) w jakości końcowych odpowiedzi:**
   - Mimo zbliżonego `recall@5` na poziomie wpisów (`0.43` dla obu strategii), `rpg_structure` osiąga wyraźnie wyższą jakość odpowiedzi ocenianą ręcznie (`human_score` **0.72** vs **0.57** dla `rpg_fixed` na poziomie `source`).
   - Chunki strukturalne zachowują pełne wpisy (zaklęcia, bloki statystyk potworów) wraz z nagłówkiem hierarchii (`książka › rozdział › sekcja › wpis`), dzięki czemu model rzadziej otrzymuje ucięte w połowie zdania lub przemieszane reguły sąsiednich wpisów. Widać to szczególnie w pytaniach kategorii `medium` (`0.70` vs `0.45`) oraz `complex` (`0.45` vs `0.30`).
   - Lepszy kontekst wiąże się z wyższym kosztem: średni rozmiar kontekstu (`ctx_chars`) w `rpg_structure` jest o ~29% większy (`5166` vs `3993`), co przekłada się na dłuższy czas generacji (`~1.8 s` vs `1.07 s`).

2. **Wąskim gardłem bazowego pipeline'u jest retrieval, a nie halucynacje modelu:**
   - Na poziomie dokładnych źródeł (`source`) `recall@5` wynosi zaledwie **0.43**, a `MRR` **0.33–0.35**. Oznacza to, że dla ponad połowy pytań oczekiwany fragment w ogóle nie trafia do `top-5`.
   - Zjawisko to wynika m.in. z nakładania się terminologii między systemami (szczególnie D&D 5e i opartego na tej samej mechanice SW5e) oraz z ograniczeń czystego wyszukiwania wektorowego (dense) przy dokładnych nazwach własnych, skrótach i tabelach.
   - Różnica między poziomem `source` a `section` (`recall@5` rośnie z `0.43` do `0.57` dla `rpg_structure` i do `0.51` dla `rpg_fixed`) pokazuje, że retrieval często trafia we właściwy rozdział lub sąsiedni wpis w tej samej sekcji, ale gubi konkretny docelowy fragment.

3. **Zachowawcza postawa modelu (`qwen3.5:4b`) – brak halucynacji kosztem fałszywych odmów:**
   - Dla pytań spoza korpusu (`unanswerable`) system osiągnął **100% skuteczności (16/16 poprawnych odmów)** w każdej konfiguracji – model nie zgaduje i nie używa wiedzy parametrycznej, gdy kontekst nie pasuje do pytania.
   - Z drugiej strony głównym źródłem błędów w całym zbiorze są **niepoprawne odmowy** (`21` w `rpg_structure` i `28–29` w `rpg_fixed`), które przeważają nad merytorycznie błędnymi odpowiedziami (`13–16` w `rpg_structure` i `15–22` w `rpg_fixed`). Gdy retrieval nie dostarczy właściwego fragmentu (lub dostarczy fragment ucięty w `rpg_fixed`), model woli bezpiecznie odmówić odpowiedzi.

4. **Rozbieżność między automatycznym `citation_hit` a `human_score`:**
   - W `rpg_structure` (`source`) model udziela poprawnych merytorycznie odpowiedzi znacznie częściej (`human_score = 0.72`) niż wskazuje na to surowy `citation_hit` (`0.34`).
   - Wynika to z faktu, że w podręcznikach RPG ta sama reguła bywa opisana w kilku miejscach (np. we wstępie podręcznika, w rozdziale ogólnym oraz w bliźniaczym systemie SW5e) – model potrafi udzielić poprawnej odpowiedzi na podstawie alternatywnego fragmentu, który nie był wpisany jako jedyny wzorzec w `golden_rpg.jsonl`.
   - Dodatkowo w `rpg_fixed` automatyczny `citation_hit` (`0.36` / `0.46`) jest nieznacznie wyższy niż w `rpg_structure` (`0.34` / `0.43`), mimo gorszej jakości samych odpowiedzi – jest to efekt uboczny faktu, że chunk o stałej długości często zahacza o granice kilku wpisów naraz i ma przypisanych więcej identyfikatorów `source_ids`.

5. **Kierunki dalszych eksperymentów (E2+):**
   - Jako bazę do kolejnych iteracji należy przyjąć kolekcję **`rpg_structure`**, ponieważ zapewnia znacznie wyższą użyteczność końcową (`human_score`).
   - Priorytetem jest podniesienie `recall@5` i `MRR` poprzez:
     - **wyszukiwanie hybrydowe (dense + BM25)** – aby poprawić dopasowanie dokładnych nazw zaklęć, mocy i terminów mechanicznych,
     - **filtrowanie metadanych (`system`) lub query rewriting** – aby zapobiec wypieraniu chunków D&D 5e przez analogiczne sekcje z SW5e,
     - **reranking (cross-encoder)** na szerszej puli kandydatów (np. `top-20` $\rightarrow$ `top-5`) – aby wyciągnąć właściwy wpis z sekcji na pierwsze pozycje.