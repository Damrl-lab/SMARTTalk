# Processed Split Schema

Each processed split file is an `.npz` archive with the following keys:

- `X`
  - dtype: `float32`
  - shape: `[num_windows, window_days, num_attributes]`
- `y`
  - dtype: integer
  - meaning: `0 = healthy`, `1 = failed / at risk`
- `ttf`
  - dtype: integer
  - meaning: time-to-failure in days
- `features`
  - dtype: string array
  - meaning: ordered SMART attribute names aligned with the last dimension of `X`

For the bundled paper setting:

- `window_days = 30`
- `num_attributes = 15`

The bundled sample files follow the same schema as the full processed data.
