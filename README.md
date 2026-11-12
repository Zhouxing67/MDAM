# Multi-Decoder Attention Model with Embedding Glimpse for Solving Vehicle Routing Problems
Liang Xin, Wen Song, Zhiguang Cao, Jie Zhang.  Multi-Decoder Attention Model with Embedding Glimpse for Solving Vehicle Routing Problems, Proceedings of the AAAI Conference on Artificial Intelligence, pp.12042-12049, 2021. [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/17430)

## Environment (Python 3.14 + uv)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:
```bash
uv sync --locked
uv run python --version
```

`pyproject.toml` declares dependencies, `.python-version` selects Python 3.14,
and `uv.lock` pins the resolved versions. uv creates the local `.venv` and
can download the required Python interpreter automatically. Use `uv add` to
add dependencies and `uv lock --upgrade` followed by `uv sync` to update them.
TensorBoard logging uses `torch.utils.tensorboard.SummaryWriter`.

The included TSP20 checkpoint is at `pretrained_tsp20/tsp_20/epoch-99.pt`.
Run shell evaluation scripts with `uv run bash eval.sh <command>` (Bash required).
The managed environment covers the MDAM training, search, data generation and
download scripts. Legacy TensorFlow code in `encode-attend-navigate`, external
solver executables, Gurobi and OR-Tools baselines require separate environments
or installation and are not part of this migration.

## Train a model

Example: TSP with 20 nodes. Training outputs are written under `outputs/`;
TensorBoard events are written under `logs/`.
```bash
uv run python run.py --graph_size 20 --baseline rollout --run_name 'tsp20' --problem="tsp" --batch_size=512 --epoch_size=1280000 --kl_loss=0.01 --n_EG=2 --n_paths=5 --val_size=10000
```

## Generate a TSP20 test dataset
```bash
mkdir data
uv run python generate_data.py --filename=data/tsp_20 --problem="tsp" --graph_sizes=20 --dataset_size=100000
```

## Evaluate the included TSP20 model

This command works in PowerShell and Bash. Add `--no_cuda` to request CPU execution.
The checkpoint must remain beside its `args.json` file.
```bash
uv run python -u search.py data/tsp_20.pkl --model=pretrained_tsp20/tsp_20/epoch-99.pt --beam_size=1 --eval_batch_size=1024
```

## Batch evaluation with Bash

These scripts require Bash and command-line tools including `unzip` and `tee`.
They use the downloaded `pretrained/` directory, which differs from the included
`pretrained_tsp20/` directory. Download and extract the full model collection
before running the problem-specific commands. The original download endpoint
has not been verified during this migration.

The scripts evaluate greedy decoding and beam widths 30 and 50:
```bash
uv run bash eval.sh download
uv run bash eval.sh generate_data
uv run bash eval.sh tsp
uv run bash eval.sh cvrp
uv run bash eval.sh sdvrp
uv run bash eval.sh op
uv run bash eval.sh pctsp
uv run bash eval.sh spctsp
```

## Environment checks and migration status

Run these checks from the repository root:

```bash
uv lock --check
uv run python --version
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
uv run python run.py --help
uv run python search.py --help
```

The project targets Python `>=3.14,<3.15`. The local migration on 2026-11-12
installed Python 3.14.7 and synchronized the locked dependencies. The lockfile,
all 63 project Python files' syntax, and `git diff --check` passed validation.

Training and inference have **not** been validated in this environment:
importing PyTorch failed on Windows with `PermissionError: [WinError 5]` while
loading `torch_python.dll` or one of its dependencies, including a retry outside
the sandbox. The cause is not yet established. If this occurs, first reproduce
it with the isolated `import torch` command above; project training and search
cannot run until the library imports successfully. Installing dependencies
successfully does not establish GPU availability or model correctness.

The migration also replaces `tensorboard_logger` with PyTorch's
`SummaryWriter` and updates the legacy `collections.Iterable` import to
`collections.abc.Iterable`. It does not resolve the pre-existing model API,
evaluation-mode, or checkpoint-saving issues identified in the repository review.

## Attribution

MDAM is developed based on https://github.com/wouterkool/attention-learn-to-route (Kool, W.; van Hoof, H.; and Welling, M. 2019. Attention,
Learn to Solve Routing Problems! In Proceedings of International Conference on Learning Representations (ICLR).)
