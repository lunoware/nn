# nn

A minimal neural network in a single C++ file. No dependencies beyond the standard library.

## Build

```bash
g++ -o nn nn.cpp -lm
```

## Usage

### Train

Reads `config.json` (architecture + hyperparameters) and `data.json` (training samples), trains the network, and saves learned weights to `weights.json`.

```bash
./nn --train
```

Example output:
```
Epoch 0 | MSE: 0.134977
Epoch 1000 | MSE: 0.002739
...
Epoch 10000 | MSE: 0.000124
Weights saved to weights.json
```

### Predict

Reads `config.json` and `weights.json`, runs inference on each row of a CSV file, and prints one prediction per line.

```bash
./nn --predict inputs.csv
```

Example output for XOR:
```
0.013589
0.984629
0.983006
0.016670
```

## Configuration

**`config.json`** — network shape and hyperparameters:
```json
{"layers":[2,4,1],"learning_rate":0.5,"epochs":10000,"seed":0}
```

- `layers` — neuron count per layer (input → hidden… → output)
- `learning_rate` — gradient descent step size
- `epochs` — number of full passes over the training data
- `seed` — random seed for weight initialisation (same seed → identical weights)

**`data.json`** — training samples:
```json
{"inputs":[[0,0],[0,1],[1,0],[1,1]],"targets":[[0],[1],[1],[0]]}
```

**`inputs.csv`** — rows to run inference on:
```
0,0
0,1
1,0
1,1
```

## Extending

Change `layers` in `config.json` to use a different architecture — no code changes needed. For example, a deeper network:

```json
{"layers":[2,3,3,1],"learning_rate":0.5,"epochs":10000,"seed":0}
```

```bash
./nn --train && ./nn --predict inputs.csv
```
