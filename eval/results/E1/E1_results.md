# E1 — podsumowanie pomiaru bazowego

## Cel i zakres

Pomiar E1 ustanawia punkt odniesienia dla bazowego pipeline'u RAG przed testowaniem
ulepszeń retrievalu. Porównano dwie kolekcje:

- `rpg_structure` — chunking strukturalny,
- `rpg_fixed` — chunking fixed-size.

Każda konfiguracja została oceniona na dwóch poziomach:

- `source` — zgodność z dokładnym `source_id` z goldena,
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

Rozmiar kontekstu (`OLLAMA_CONTEXT_LENGTH`) został ustawiony na 16384.

## Zbiór testowy

Źródło: [`golden_rpg.jsonl`](../../golden_rpg.jsonl)

- 120 pytań,
- 104 pytania z oczekiwaną odpowiedzią w korpusie,
- 16 pytań spoza korpusu, dla których oczekiwana jest odmowa,
- podręczniki reprezentowane w goldenie:
  - `PlayerDnDBasicRules.pdf` — 47 odwołań,
  - `Mythras - 2E (v3.1).pdf` — 30,
  - `SW5e - Player's Handbook.pdf` — 21,
  - `DMBasicRulesv.pdf` — 14.

Zbiór obejmuje pytania o reguły D&D 5e, Mythras i SW5e, retrieval
cross-system oraz pytania na które nie ma odpowiedzi w korpusie z innych systemów i tematów.

## Wyniki

Szczegółowe wyniki znajdują się w katalogu [`eval/results/E1`](./).

### Retrieval

| Kolekcja | Poziom | recall@1 | recall@3 | recall@5 | MRR | ctx_chars |
|---|---:|---:|---:|---:|---:|---:|
| `rpg_fixed` | source | **0.27** | **0.36** | **0.43** | **0.33** | **3993** |
| `rpg_fixed` | section | **0.30** | **0.44** | **0.51** | **0.40** | **3993** |
| `rpg_structure` | source | **0.29** | **0.39** | **0.43** | **0.35** | **5166** |
| `rpg_structure` | section | **0.34** | **0.52** | **0.57** | **0.45** | **5166** |


### Generowanie

| Kolekcja | Poziom | refusal_ok | citation_hit | inline_cited | latency |
|---|---:|---:|---:|---:|---:|
| `rpg_structure` | source | **0.85** | **0.34** | **0.72** | **1.83 s** |
| `rpg_structure` | section | **0.85** | **0.43** | **0.71** | **1.79 s** |
| `rpg_fixed` | source | **0.86** | **0.36** | **0.77** | **1.07 s** |
| `rpg_fixed` | section | **0.88** | **0.46** | **0.80** | **1.07 s** |

`latency` obejmuje retrieval i generowanie. `citation_hit` sprawdza, czy wśród
cytowanych chunków znalazło się oczekiwane źródło, a `inline_cited` — czy odpowiedź
zawierała znaczniki `[n]`.

Szczegółowe tabele per pytanie znajdują się w plikach:

- [`e1_rpg_structure_qwen3.5-4b.md`](./e1_rpg_structure_qwen3.5-4b.md),
- [`e1_rpg_structure_section_qwen3.5-4b.md`](./e1_rpg_structure_section_qwen3.5-4b.md),
- [`e1_rpg_fixed_qwen3.5-4b.md`](./e1_rpg_fixed_qwen3.5-4b.md),
- [`e1_rpg_fixed_section_qwen3.5-4b.md`](./e1_rpg_fixed_section_qwen3.5-4b.md).
