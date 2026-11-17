# idm-ttrpg – RAG dla podręczników RPG

Szablon projektu z kursu **Inżynieria dużych modeli językowych** (WEiTI PW) dostosowany do **podręczników gier fabularnych (TTRPG)**.
Minimalny, czytelny pipeline RAG dla Mistrza Gry i graczy, który uruchomisz lokalnie na laptopie:

```
 ingest (offline)                                          query (online)
 ─────────────────                                         ───────────────
 PDF/MD/TXT → normalizacja (ligatury) → chunking RPG ─┐    pytanie (+ filtr systemu) → embedding → retrieval top-k
   (outline/TOC → rozdziały/sekcje + zaklęcia/potwory) ▼                                             │
                                             Qdrant (wektory + metadane RPG) ◄───────────────────────┘
                                                                                                     ▼
                                       odpowiedź MG z cytowaniem (np. PHB, s. 90) ◄── LLM (structured output)
```

- **Cztery podręczniki RPG** w `data/raw/` (poza gitem, opisane w `data/raw/manifest.json` z sumami SHA-256):
  - **D&D 5e Player's Basic Rules** (`PHB`) – zasady gracza i 151 zaklęć,
  - **D&D 5e DM's Basic Rules** (`DMG`) – zasady prowadzenia i 205 stat blocków potworów,
  - **Mythras 2E** (`Mythras`) – mechanika d100, walka i magia (z korektą przesunięcia numeracji stron),
  - **Star Wars 5e Player's Handbook** (`SW5e`) – konwersja 5e ze spisem treści parsowanym ze stron 3–4, naprawą uszkodzonych ligatur `U+0000` i 385 mocami Mocy/Technologii.
- **Chunking strukturalny (`src/rag/rpg.py`)** – buduje hierarchię rozdziałów i sekcji z zakładek PDF (`outline`) lub tekstu spisu treści (`TABLE OF CONTENTS`), wycina żywą paginę i wykrywa pojedyncze wpisy: zaklęcia (`spell`), moce (`power`) oraz stat blocki potworów (`monster`).
- **Identyfikatory niezależne od strategii chunkingu (`source_ids`)** – np. `PlayerDnDBasicRules.pdf#chapter-11-spells/fireball`, co pozwala uczciwie porównać chunking strukturalny (`structure`) z podziałem na stałe okna (`fixed`/`recursive`) na poziomie wpisu (`--level source`) oraz sekcji (`--level section`).
- LLM i embeddingi przez SDK `openai` z `base_url` z `.env` – lokalna **Ollama** albo **Gemini API** (zmiana tylko w `.env`). Bez LangChaina/LlamaIndexa.
- Wektory i indeksy payloadu (`system`, `role`, `content_type`, `chapter`, `section`, `entry`) w **Qdrancie**.

## Spis treści

1. [Wymagania sprzętowe](#wymagania-sprzętowe)
2. [Instalacja](#instalacja)
3. [Okno kontekstu Ollamy – koniecznie przeczytaj](#okno-kontekstu-ollamy--koniecznie-przeczytaj)
4. [Szybki start](#szybki-start)
5. [Podręczniki RPG i chunking strukturalny](#podręczniki-rpg-i-chunking-strukturalny)
6. [Konfiguracja (.env)](#konfiguracja-env)
7. [Structured output: trzy tryby](#structured-output-trzy-tryby)
8. [Słabszy laptop: Gemini API](#słabszy-laptop-gemini-api)
9. [Ewaluacja i ręczna ocena odpowiedzi](#ewaluacja-i-ręczna-ocena-odpowiedzi)
10. [Co dalej (TODO)](#co-dalej-todo)
11. [Struktura repozytorium](#struktura-repozytorium)
12. [Troubleshooting](#troubleshooting)

## Wymagania sprzętowe

Wybierz profil pasujący do Twojego laptopa i przepisz jego ustawienia do `.env`
(`OLLAMA_CONTEXT_LENGTH` ustawiasz w systemie – patrz [okno kontekstu](#okno-kontekstu-ollamy--koniecznie-przeczytaj)).

| Profil | RAM | `LLM_MODEL` | `OLLAMA_CONTEXT_LENGTH` | `TOP_K` / `CHUNK_SIZE` | Uwagi |
|--------|-----|-------------|-------------------------|------------------------|-------|
| **S** | 8 GB | `qwen3.5:4b` (awaryjnie `qwen3.5:2b`) | 8192 | 4 / 600 | zamknij przeglądarkę i IDE z wtyczkami AI; Qdrant w trybie `local:` oszczędza ~0,5 GB |
| **M** | 16 GB | `qwen3.5:9b` | 16384 | 5 / 800 | wyraźnie lepsze odpowiedzi, ok. 1,5–2× wolniej niż 4b |
| **L** | 24 GB+ | `qwen3.5:9b` (porównania z 4b) | 16384 | 5 / 800 | oba modele mieszczą się naraz w pamięci – wygodne do eksperymentów |
| **API** | dowolny | Gemini | – | 5 / 800 | gdy laptop nie uciągnie modelu – patrz [niżej](#słabszy-laptop-gemini-api) |

Pamięć (okno 16384, `ollama ps`): `qwen3.5:4b` ≈ 4 GB, `qwen3.5:9b` ≈ 7,6 GB, embeddingi ≈ 0,6 GB.
Dysk: 5 GB (4b) lub 9 GB (9b) na modele i ~1 GB na zależności Pythona.

## Instalacja

Potrzebujesz trzech rzeczy: **uv** (Python i zależności), **Ollama** (modele lokalnie)
i opcjonalnie **Docker** (Qdrant). Nic nie instalujemy globalnie poza tymi narzędziami –
`uv` sam pobierze Pythona 3.12 do swojego katalogu i utworzy `.venv` w repozytorium.

### macOS

```bash
brew install uv ollama
brew services start ollama          # albo aplikacja Ollama.app
# Docker: OrbStack (lekki, polecany) lub Docker Desktop
brew install --cask orbstack
```

Na Macu Ollama musi działać **natywnie**, nie w Dockerze – kontener nie ma dostępu do GPU (Metal),
więc model działałby kilka-kilkanaście razy wolniej.

### Linux

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh     # instaluje też usługę systemd
# Docker: pakiet docker.io / docker-ce + plugin compose, użytkownik w grupie docker
```

### Windows

```powershell
winget install --id=astral-sh.uv -e
winget install --id=Ollama.Ollama -e
winget install --id=Docker.DockerDesktop -e     # opcjonalnie
```

`make` na Windowsie zwykle nie jest dostępny – zamiast `make ask Q="..."` wpisz
`uv run rag ask "..."` (każda komenda w `Makefile` to tylko taki wrapper; `make help` pokazuje listę).
W PowerShellu przed uruchomieniem ustaw kodowanie UTF-8:
```powershell
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
```

### Bez Dockera

Ustaw w `.env`: `QDRANT_URL=local:./.qdrant`. Qdrant działa wtedy jako biblioteka w procesie
Pythona i trzyma dane w katalogu `.qdrant/`. Wystarczy do projektu, ale: tylko jeden proces naraz
(nie uruchamiaj równolegle `ingest` i `ask`), brak dashboardu, wolniej przy dużych korpusach.

## Okno kontekstu Ollamy – koniecznie przeczytaj

**Okno kontekstu** to liczba tokenów, które model widzi naraz (instrukcje, chunki, pytanie,
odpowiedź). Ollama domyślnie ustawia małe okno (często 2048–4096) i dłuższy prompt **obcina po
cichu, bez błędu** – model nie widzi części chunków, a Ty szukasz błędu w retrievalu, którego nie ma.
Pięć chunków strukturalnych (do `1.5 × CHUNK_SIZE`) to ok. 2500–3000 tokenów, dlatego ustawiamy okno jawnie:

| System | Jak ustawić (potem zrestartuj Ollamę) |
|--------|---------------------------------------|
| macOS (aplikacja) | `launchctl setenv OLLAMA_CONTEXT_LENGTH 16384` |
| macOS (brew)      | `brew services stop ollama`, potem `OLLAMA_CONTEXT_LENGTH=16384 ollama serve` w osobnym terminalu |
| Linux (systemd)   | `sudo systemctl edit ollama` → `[Service]` `Environment="OLLAMA_CONTEXT_LENGTH=16384"` → `sudo systemctl restart ollama` |
| Windows           | `setx OLLAMA_CONTEXT_LENGTH 16384`, zamknij Ollamę z zasobnika i uruchom ponownie (`taskkill /F /IM ollama.exe /T`; `taskkill /F /IM "ollama app.exe" /T`) |

Sprawdzenie: `ollama ps` (kolumna CONTEXT) albo `make check` (krok „Okno kontekstu” ostrzega, gdy okno < 8192).

## Szybki start

```bash
make setup       # uv sync + utworzenie .env z .env.example
make pull        # ollama pull modeli z .env (Windows: make pull_w)
make up          # Qdrant w Dockerze (pomiń, jeśli QDRANT_URL=local:...)
make check       # smoke test – wszystko powinno być ✅
make rulebooks   # pobiera podręczniki RPG do data/raw
make ingest-rpg  # indeksuje 4 podręczniki z data/raw w kolekcjach rpg_structure i rpg_fixed
make ask Q="Jaki zasięg i obrażenia ma czar Fireball w D&D 5e?" ARGS='--filter "system=D&D 5e"'
make agent Q="Porównaj jak działa pancerz w D&D 5e oraz w systemie Mythras."
make eval-rpg    # porównuje rpg_structure i rpg_fixed na eval/golden_rpg.jsonl (source + section)
```

Jeśli nie masz jeszcze pobranych PDF-ów w `data/raw/`, możesz przetestować pipeline na krótkich plikach przykładowych w `data/sample/`:
```bash
uv run rag ingest data/sample --collection sample --recreate
uv run rag eval --golden eval/golden_sample.jsonl --collection sample
```

Przykładowy wynik `uv run rag ask "Jaki zasięg i obrażenia ma czar Fireball w D&D 5e?" --filter "system=D&D 5e"`:

```
Czar Fireball (Kula Ognia) w D&D 5e ma zasięg 150 stóp (150 feet) (PHB, s. 90). Cel w obszarze
działania otrzymuje 8k6 obrażeń od ognia (fire damage) przy nieudanym rzucie obronnym na Zręczność
(Dexterity saving throw) lub połowę tych obrażeń przy udanym rzucie (PHB, s. 90).

Źródła cytowane:
  [1] D&D 5e Player's Basic Rules › Chapter 11: Spells › Fireball (s. 90)  (score 0.812)
      „A target takes 8d6 fire damage on a failed save, or half as much damage on a successful one.”
retrieval 0.14s · generacja 2.40s · structured output: native
```

Znaczniki `[n]` w tekście wstawia model, a `render_answer` zamienia je na czytelne odwołania `(PHB, s. 90)` z wydrukowanym numerem strony z podręcznika (`page_label`).

## Podręczniki RPG i chunking strukturalny

Plik `data/raw/manifest.json` definiuje metadane każdego podręcznika (`system`, `edition`, `role`, `language`, `label`, `short`, `doc_type: "rpg"`) oraz sumę `sha256`. Podczas `ingest`:

1. **Normalizacja (`src/rag/normalize.py`)** – oprócz standardowych ligatur Unicode i sklejania wyrazów naprawia specyficzny defekt fontu w *SW5e Player's Handbook*, gdzie wszystkie ligatury (`fi`, `fl`, `ff`, `ffi`, `ffl`) są zastąpione znakiem `U+0000` (np. `modi\x00er` → `modifier`, `e\x00ect` → `effect`, `\x00ight` → `flight`, `mblu\x00ng` → `mbluffing`).
2. **Struktura (`src/rag/rpg.py`)**:
   - Czyta drzewo zakładek PDF (`read_outline`), a dla książek bez zakładek (SW5e) parsuje tekst spisu treści (`parse_toc`) z zachowaniem wcięć.
   - Usuwa żywą paginę i wykrywa wydrukowany numer strony (`page_label`), korygując przesunięcie numeracji (np. w *Mythras 2E* wydrukowana strona = fizyczna − 1).
   - Wykrywa zaklęcia (`spell`), moce (`power`) i stat blocki potworów (`monster`), tworząc dokładne `source_ids`, np. `DMBasicRulesv.pdf#monsters-a-z/owlbear`.
   - W strategii `structure` każdy chunk otrzymuje nagłówek kontekstowy (`Książka › Rozdział › Sekcja › Wpis`) i limit `1.5 × CHUNK_SIZE`, dzięki czemu stat blocki i opisy zaklęć nie są cięte w połowie.

## Konfiguracja (.env)

Wszystkie ustawienia są w `.env` (szablon: `.env.example`), wczytuje je `src/rag/config.py`:

| Zmienna | Domyślnie | Opis |
|---------|-----------|------|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Ollama, `qwen3.5:4b` | serwer zgodny z OpenAI |
| `LLM_REASONING_EFFORT` | `none` | `none` wyłącza tryb thinking |
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | model embeddingów |
| `DATA_DIR` | `data/raw` | katalog z podręcznikami dla `make ingest` |
| `QDRANT_URL`, `COLLECTION` | `http://localhost:6333`, `rpg_structure` | vector store i domyślna kolekcja |
| `TOP_K`, `CHUNKING`, `CHUNK_SIZE`, `CHUNK_OVERLAP` | 5, `structure`, 800, 120 | parametry RAG (`structure` / `recursive` / `fixed`) |
| `LLM_TEMPERATURE` | `0` | temperatura generacji (0 = powtarzalne odpowiedzi i eval) |
| `EMBED_QUERY_INSTRUCTION` | instrukcja RPG dla qwen3-embedding | doklejana do zapytań (nie do dokumentów) |
| `STRUCTURED_OUTPUT` | `auto` | `auto` / `native` / `tool` / `prompt` |

## Structured output: trzy tryby

Odpowiedź RAG to JSON zgodny ze schematem `Answer` (`src/rag/generate.py`): `answer` (tekst ze
znacznikami `[n]`) i `citations` (numery fragmentów; pusta lista = odmowa). `src/rag/structured.py`
obsługuje trzy tryby: **native** (`response_format`), **prompt** (schemat w prompcie + walidacja Pydantic)
oraz **tool** (`STRUCTURED_OUTPUT=tool`). Domyślny tryb `auto` zapisuje wykryty wariant w `.cache/structured_output.json`.

## Słabszy laptop: Gemini API

Odkomentuj blok Gemini w `.env` (szablon w `.env.example`): klucz z
[Google AI Studio](https://aistudio.google.com/apikey), `EMBED_MODEL=gemini-embedding-001`, pusta
`EMBED_QUERY_INSTRUCTION`, potem `make ingest-rpg`. Qdrant nadal działa lokalnie.

## Ewaluacja i ręczna ocena odpowiedzi

Złoty zbiór `eval/golden_rpg.jsonl` zawiera pytania przekrojowe dla wszystkich czterech podręczników (zaklęcia, potwory, mechanika walki, moce SW5e oraz pytania spoza korpusu sprawdzające odmowę odpowiedzi). Każdy rekord posiada dokładne `source_ids` (poziom wpisu) oraz `section_ids` (poziom sekcji/rozdziału):

```bash
make eval-rpg                        # porównanie rpg_structure vs rpg_fixed (poziom source i section)
make eval-rpg ARGS=--generate        # + ocena wygenerowanych odpowiedzi, odmów i cytowań
```

### Ręczna ocena odpowiedzi (`rag review`)

Po uruchomieniu ewaluacji z flagą `--generate` możesz przejrzeć i ręcznie ocenić odpowiedzi modelu w terminalu (poprawna / niepoprawna / poprawna odmowa / błędna odmowa):

```bash
uv run rag review eval/results/<plik_wynikow>.json
uv run rag review eval/results/<plik_wynikow>.json --only-ungraded
```

Oceny są zapisywane bezpośrednio w pliku JSON (`human_label`, `human_score`, `human_note`).
`review` wyświetla i zapisuje w pliku JSON oraz odpowiadającym mu pliku `.md`
(`summary.human_review`) liczbę ocenionych oraz pominiętych pytań, liczności i udziały etykiet
(`correct`, `incorrect`, `refusal_ok`, `refusal_bad`) oraz metrykę `human_score`. To samo
podsumowanie jest generowane osobno dla każdego tagu w polu `summary.human_review.by_tag`.
Przy wielu tagach pytanie jest przypisywane wyłącznie do pierwszego tagu.

## Podsumowanie pomiaru bazowego E1
[`eval/results/E1/E1_results.md`](eval/results/E1/E1_results.md).

## Co dalej (TODO)

W `src/rag/retrieve.py` znajdują się miejsca na dalsze rozszerzenia – po każdej zmianie uruchom `make eval-rpg`:

| TODO | Gdzie | Idea |
|------|-------|------|
| `TODO(hybrid)` | `hybrid_search()`, `store.py` | dense + sparse BM25 (FastEmbed `Qdrant/bm25`), fuzja RRF – szczególnie skuteczna dla dokładnych nazw własnych zaklęć i potworów |
| `TODO(reranking)` | `rerank()` | cross-encoder `BAAI/bge-reranker-v2-m3` na top-20 kandydatach |
| `TODO(metadata-filtering)` | `build_filter()` | rozbudowa filtrów payloadu (`MatchAny`, zakresy stron) |
| `TODO(query-rewriting)` | `rewrite_query()` | tłumaczenie polskich zapytań gracza na angielskie terminy z podręczników (cross-lingual query expansion) |

## Struktura repozytorium

```
src/rag/   jeden moduł = jeden krok pipeline'u (rpg.py – parser i chunking podręczników RPG)
data/      sample/ – przykładowe pliki RPG (MD/TXT) · raw/ – 4 podręczniki PDF + manifest.json
eval/      golden_rpg.jsonl i golden_sample.jsonl · results/ – wyniki i comparison.md
tests/     testy jednostkowe bez sieci i bez PDF-ów (make test)
scripts/   pomocnicze skrypty (pobieranie modeli)
```

## Troubleshooting

| Objaw | Przyczyna / rozwiązanie |
|-------|-------------------------|
| `❌ Ollama działa` | Ollama nie jest uruchomiona: `ollama serve` / aplikacja. |
| `❌ Modele pobrane` | `make pull` (lub `make pull_w` na Windowsie). |
| `❌ Qdrant działa` | `make up` albo ustaw `QDRANT_URL=local:./.qdrant`. |
| `Kolekcja ... nie istnieje` | najpierw `make ingest-rpg` (lub `make ingest`). |
| `UnicodeEncodeError` w PowerShellu | ustaw `$env:PYTHONUTF8='1';$env:PYTHONIOENCODING='utf-8'` przed uruchomieniem `uv run rag ...`. |
| `make: command not found` (Windows) | używaj bezpośrednio `uv run rag <komenda>` (patrz `Makefile`). |


