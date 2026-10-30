"""Generate an educational synthetic component-aging time-series dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

def generate_dataset(output_path="data/component_aging.csv", n_components=180, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for idx in range(n_components):
        component_id = f"C{idx+1:04d}"
        # Component-to-component variation in baseline and aging sensitivity.
        rated_life = int(rng.integers(650, 1450))
        sensitivity = rng.uniform(0.75, 1.30)
        base_esr = rng.uniform(18, 34)
        base_cap = rng.uniform(880, 1120)
        baseline_eff = rng.uniform(91.0, 97.0)
        cycles = np.arange(0, rated_life + 1, 25)
        if cycles[-1] != rated_life:
            cycles = np.append(cycles, rated_life)
        for cycle in cycles:
            ambient = rng.normal(27, 4)
            load = np.clip(rng.normal(68, 16), 15, 100)
            voltage = rng.normal(12.0, 0.35)
            ripple = np.clip(rng.normal(35 + 0.035*cycle, 8), 5, 140)
            vibration = np.clip(rng.normal(0.12 + 0.00012*cycle, 0.045), 0.01, 0.9)
            case_temp = ambient + 0.19*load + rng.normal(0, 2.0)
            # Temperature and load accelerate degradation in this simplified simulator.
            stress = max(0.1, 1 + 0.018*(case_temp-45) + 0.006*(load-60))
            degradation = sensitivity * (cycle / rated_life)**1.45 * stress
            degradation = max(0.0, degradation + rng.normal(0, 0.025))
            esr = base_esr * (1 + 1.25*degradation) + 0.07*cycle/rated_life*base_esr + rng.normal(0, 1.2)
            capacitance = base_cap * (1 - 0.22*degradation) + rng.normal(0, 9)
            efficiency = baseline_eff - 5.5*degradation - 0.025*max(0, case_temp-40) + rng.normal(0, 0.7)
            eol_cycle = rated_life
            rul = max(0, int(eol_cycle - cycle))
            failed = int(cycle >= eol_cycle)
            rows.append({
                "component_id": component_id, "cycle": int(cycle),
                "ambient_temp_c": round(float(ambient), 3),
                "case_temp_c": round(float(case_temp), 3),
                "load_pct": round(float(load), 3),
                "input_voltage_v": round(float(voltage), 3),
                "ripple_mv": round(float(ripple), 3),
                "vibration_g": round(float(vibration), 4),
                "esr_mohm": round(float(esr), 3),
                "capacitance_uf": round(float(capacitance), 3),
                "efficiency_pct": round(float(efficiency), 3),
                "degradation_index": round(float(degradation), 5),
                "failed": failed, "rul_cycles": rul,
                "dataset_type": "synthetic_simulation"
            })
    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    df = generate_dataset()
    print(f"Saved {len(df):,} rows to data/component_aging.csv")
    print(df.head().to_string(index=False))
