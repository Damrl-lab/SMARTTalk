# Data

This folder collects the data-related pieces of the repository.

It includes:

- `sample_data/` with small `.npz` files for smoke tests,
- `processed_schema.md` describing the processed split format,
- `raw/` with instructions for placing the public Alibaba source data,
- `splits/` as the output location for generated MB1 / MB2 train, validation,
  and test splits.

The full raw dataset is not bundled here. Download and placement instructions
are documented in `../DATA_ACCESS.md`.

The large processed arrays (the full original `splits/` and the large sampled MB2
`splits_sampled/MB2_round*/train.npz`) are not stored in git; download them from
Google Drive and place the `splits/` and `splits_sampled/` folders under `data/`:
<https://drive.google.com/drive/folders/1oMqZz4nr5Q071f20i2dCbvPb3gaoPn2x?usp=sharing>
