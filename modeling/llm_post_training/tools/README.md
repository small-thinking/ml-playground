# Vocabulary Inspector

Simple tool to inspect HuggingFace model vocabularies and find similar tokens.

## Usage

```bash
# Find similar tokens
python vocab_inspect.py --model-path distilbert-base-uncased --query "dog" --top-k 5

# List vocabulary tokens
python vocab_inspect.py --model-path distilbert-base-uncased --list-tokens --limit 20
```

## Examples

```bash
# Find similar tokens with compact display
$ python vocab_inspect.py --model-path distilbert-base-uncased --query "dog" --top-k 3
Similar to 'dog':
Query: dog (id:3899, len:3, alpha:True)
 1. dog              1.000 (id:3899, len:3, alpha:True)
 2. ##dog            0.684 (id:16168, len:5, alpha:False)
 3. dogg             0.645 (id:28844, len:4, alpha:True)

# List tokens
$ python vocab_inspect.py --model-path distilbert-base-uncased --list-tokens --limit 5
Tokens:
  1. ##tosis
  2. ##dev
  3. [unused405]
  4. ghetto
  5. ##oya
(5 tokens)
```

## Options

| Option            | Description                             | Default |
| ----------------- | --------------------------------------- | ------- |
| `--model-path`    | HuggingFace model path (required)       | -       |
| `--query`         | Find similar tokens                     | -       |
| `--top-k`         | Number of similar tokens                | 10      |
| `--method`        | Similarity method (auto/embedding/text) | auto    |
| `--list-tokens`   | List vocabulary tokens                  | False   |
| `--limit`         | Limit tokens to list                    | None    |
| `--pattern`       | Filter tokens by pattern                | None    |
| `--device`        | Device (auto/cuda/cpu)                  | auto    |
| `--no-embeddings` | Skip model loading                      | False   |
