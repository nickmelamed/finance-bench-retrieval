from pathlib import Path
import orjson


def save_json(data, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def load_json(path: str):
    with open(path, "rb") as f:
        return orjson.loads(f.read())