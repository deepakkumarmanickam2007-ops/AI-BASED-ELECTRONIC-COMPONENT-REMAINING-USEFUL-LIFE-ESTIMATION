# Dataset feature dictionary

Each row represents one observation of one simulated electronic component at a point in its operating life.

| Column | Meaning | Unit / type |
|---|---|---|
| `component_id` | Unique simulated component identifier | categorical ID |
| `cycle` | Age of the component at observation | simulated cycles |
| `ambient_temp_c` | Ambient temperature | °C |
| `case_temp_c` | Estimated component case temperature | °C |
| `load_pct` | Load level | % |
| `input_voltage_v` | Input voltage | V |
| `ripple_mv` | Supply ripple amplitude | mV |
| `vibration_g` | Vibration magnitude | g |
| `esr_mohm` | Equivalent series resistance proxy | mΩ |
| `capacitance_uf` | Capacitance proxy | µF |
| `efficiency_pct` | Estimated conversion efficiency | % |
| `degradation_index` | Simulated normalized degradation index | 0–1+ |
| `failed` | Whether the simulated EOL threshold has been reached | 0/1 |
| `rul_cycles` | Remaining simulated cycles to EOL | target |
| `dataset_type` | Provenance label | `synthetic_simulation` |

The data is generated using a stochastic educational simulator, not collected from real hardware.
