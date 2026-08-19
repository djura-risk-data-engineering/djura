The slow tests need a local flatfile; additional flatfiles must be provided
by the user and pointed at with ``DJURA_METADATA_PATH``.

```sh
set DJURA_METADATA_PATH=C:\path\to\flatfile.pickle

pytest --cov=src -cov-report=html

coverage html
```
