# Edge-Efficient Reasoner — Low-Rank Adapter Construction

This repository implements efficient low-rank adapters for Boolean reasoning, as formalized in Addenda A1, A2, and A3 of the associated proof (see `proof.md` for details). The code provides a simple, device-agnostic pipeline for building and verifying adapters on binary data.

## Notation (↔ proof reference)

| Symbol | Meaning                                 | Proof Reference |
|--------|-----------------------------------------|----------------|
| X      | Stacked hidden states                   | A2-R           |
| Y      | ±1 label matrix                         | A1-T           |
| U      | (XᵀX)⁻¹XᵀY (A1-T separator)             | A1-T separator |
| V      | d×r basis whose first r rows are I_r    | A3-P requirement|
| δ      | U Vᵀ   (low-rank adapter)               |                |

_All tensors are placed on the device returned by `choose_device()`._

## Usage

### Requirements

- Python 3.8+
- PyTorch
- NumPy

### Running the Unit Test

```
python edge_efficient_reasoner.py
```

The unit test verifies rank properties and Boolean recovery for a 2-atom truth table.

### Building an Adapter

The main pipeline is in `build_adapter(X, Y)`, which expects:

- `X`: (m, d) tensor of hidden states, full column-rank
- `Y`: (m, r) tensor of ±1 labels

It returns a low-rank adapter matrix δ.

### Boolean Decoding

The function `boolean_decode(U, h, Vt=None)` applies the learned separator with optional basis transformation.

### Bias Handling

To use a bias (shifted threshold), append a column of 1s to your `X` before training and to your input `h` at decode time. This enables Boolean decoding via `logits >= 0`.

## Directory Structure

```
.
├── edge_efficient_reasoner.py
├── README.md
└── proof.md
    
```

## Proof

See the `proof.md` document for the formal proof and addenda referenced in the code and notation above.

## License

MIT License

## Acknowledgments

- Addenda A1, A2, and A3 formal proof design
- Device selection logic inspired by codeEx2 guidelines
