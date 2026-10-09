# Dataset metas

The `train_metas_path` used by `train_simvla2.py` is a JSON file describing
where each episode of a dataset lives on disk. This directory ships one
meta per benchmark:

| File                 | Benchmark    | Placeholder         | Format                                              |
| -------------------- | ------------ | ------------------- | --------------------------------------------------- |
| `bridge.json`        | Bridge V2    | `<BRIDGE_V2_ROOT>`  | TFDS root (single `data_dir`)                       |
| `bridge_lang.json`   | Bridge V2    | -- (no paths)       | instruction paraphrases for language augmentation   |
| `fractal.json`       | Fractal      | `<FRACTAL_ROOT>`    | TFDS root (single `data_dir`)                       |
| `libero.json`        | LIBERO       | `<LIBERO_ROOT>`     | 130 HDF5 paths + `data_dir`                         |
| `calvin.json`        | CALVIN       | `<CALVIN_ROOT>`     | 17 870 per-episode HDF5 paths                       |

## Point the metas at your data

Every meta uses a `<NAME_ROOT>` placeholder in place of the training-node
absolute path. Two options to make them resolvable:

1. **Rewrite in place with sed** — one command per meta:
   ```bash
   sed -i "s|<BRIDGE_V2_ROOT>|/path/to/your/bridge_data_v2|g" datasets/metas/bridge.json
   sed -i "s|<FRACTAL_ROOT>|/path/to/your/fractal20220817_data|g" datasets/metas/fractal.json
   sed -i "s|<LIBERO_ROOT>|/path/to/your/libero|g" datasets/metas/libero.json
   sed -i "s|<CALVIN_ROOT>|/path/to/your/calvin|g" datasets/metas/calvin.json
   ```

2. **Symlink your data at those placeholder names** — if you want the shipped
   metas to keep working as-is across machines, create matching directories:
   ```bash
   mkdir -p $HOME/simvla2-data
   ln -s /your/bridge_data_v2  $HOME/simvla2-data/BRIDGE_V2_ROOT
   ln -s /your/libero          $HOME/simvla2-data/LIBERO_ROOT
   # ...etc
   sed -i "s|<\([A-Z0-9_]*\)>|$HOME/simvla2-data/\1|g" datasets/metas/*.json
   ```

The LIBERO meta covers all five suites, including `libero_90`. Each shipped
meta is meant to be used as-is: edit the placeholder as above rather than
regenerating it.
