```sh
set DJURA_METADATA_PATH=src/djura/record_selection/assets/NGA_W2_v2.pickle

pytest --cov=src -cov-report=html

coverage html
```
