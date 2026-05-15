# E.coli UTI Disease Demo

These files were recovered from the live server demo project and kept as a small,
GitHub-safe workflow sample.

- `for_ecoli_lab_test.csv` - peptide prediction table used as simulation input
- `ecoli_disease_plugin.json` - disease plugin consumed by `labsim_demo.py`
- `ecoli_lab_sim.csv` - reference lab simulation output
- `simulation_2.csv` - second reference simulation output from the server

Run locally from the repository root:

```bash
source .venv/bin/activate
python labsim_demo.py \
  --peptides_csv demo/ecoli_uti/for_ecoli_lab_test.csv \
  --plugin demo/ecoli_uti/ecoli_disease_plugin.json \
  --out_csv /tmp/evotensor_ecoli_lab_sim.csv
```

