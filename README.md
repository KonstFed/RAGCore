# RAG Core

Monorepo для проекта RAGCore.

## Структура проекта

```
RAGCore/
├── packages/
│   └── repochunker/          # Пакет для чанкирования репозиториев
│       ├── src/
│       │   └── repochunker/
│       │       ├── repoiter.py      # Итератор по файлам репозитория
│       │       ├── chunker.py      # Роутер для чанкирования
│       │       ├── models.py       # Pydantic модели
│       │       └── config_utils.py  # Утилиты для работы с конфигурацией
│       └── pyproject.toml
└── pyproject.toml            # Конфигурация workspace
```

## Установка

Установите [uv](https://github.com/astral-sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Установка зависимостей

```bash
uv sync
```

Эта команда установит все зависимости и workspace-пакеты в editable режиме.

### Использование

Быстрая проверка что работает

```bash
cd packages/repochunker/src/repochunker
uv run python chunker.py
```

```python
from repochunker.chunker import RouterChunkerConfig
from repochunker.config_utils import load_config

config = load_config(RouterChunkerConfig, "path/to/config.yaml")
router = config.create()
chunks = router.chunk_repo(Path("/path/to/repo"))
```

