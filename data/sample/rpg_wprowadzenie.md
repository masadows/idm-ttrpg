# RAG dla podręczników RPG

## Po co RAG w RPG

Podręcznik RPG ma setki stron, a odpowiedź na pytanie przy stole często zależy od jednego zaklęcia,
stat blocku albo wyjątku w zasadach. RAG (Retrieval-Augmented Generation) najpierw wyszukuje
najbardziej pasujące fragmenty, a dopiero potem podaje je modelowi. Dzięki temu model nie musi
pamiętać całej książki, a Ty możesz sprawdzić, z którego fragmentu pochodzi odpowiedź.

## Jak działa retrieval

1. Dokument jest dzielony na chunki i zamieniany na embeddingi.
2. Pytanie także dostaje embedding.
3. Qdrant zwraca top-k chunków o największym podobieństwie.
4. Chunki trafiają do promptu razem z instrukcją i numerami `[1]`, `[2]`, ...

Retrieval nie „rozumie” zasad tak jak człowiek. Dobiera fragmenty po podobieństwie semantycznym,
więc pytanie „co robi fireball” może trafić w opis zaklęcia, ale pytanie o wyjątek w walce
może wymagać lepszego zapytania albo większego `TOP_K`.

## Cytowania i źródła

Model zwraca numery fragmentów, np. `[2]`. Kod zamienia je na czytelne źródło:
`D&D 5e – Player's Basic Rules, s. 90`. Model nie przepisuje nazwy pliku ani numeru strony,
więc nie może ich wymyślić. Cytowanie jest ważne w RPG: dzięki niemu Mistrz Gry może szybko
sprawdzić, czy odpowiedź nie jest interpretacją zamiast zasadą.

## Ograniczenia

- RAG nie zastępuje podręcznika – pokazuje fragmenty, a nie pełny kontekst.
- Jeden chunk może zawierać kilka zasad, jeśli podział książki nie jest wystarczająco drobny.
- Przy pytaniach o kilka dokumentów retrieval może pominąć jedno źródło.
- Model nadal może błędnie zinterpretować poprawnie znaleziony fragment.

Dlatego eval mierzy osobno retrieval (recall@k, MRR) i jakość odpowiedzi (cytowania, odmowy).
