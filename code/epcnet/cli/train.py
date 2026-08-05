import argparse
import logging

from torch.utils.data import DataLoader

from epcnet.data.dataset import PairedSignalDataset, collate_signals
from epcnet.data.sampling import BatchEffectAwareSampler
from epcnet.losses.composite import EPCObjective
from epcnet.models.network import EPCNet
from epcnet.randomness import set_seed
from epcnet.settings import load_config
from epcnet.training.distributed import finalize, initialize, wrap
from epcnet.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="epc-train")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    context = initialize()
    set_seed(args.seed + context.rank)
    dataset = PairedSignalDataset(args.data)
    sampler = BatchEffectAwareSampler(
        dataset.cohorts,
        dataset.instruments,
        dataset.acquisition_dates,
        config.train.batch_size,
        args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_signals,
        num_workers=config.train.workers,
        pin_memory=context.device.type == "cuda",
        persistent_workers=config.train.workers > 0,
    )
    model = EPCNet(config)
    model = wrap(model, context)
    objective = EPCObjective(config.loss)
    trainer = Trainer(model, objective, config, context, loader, args.seed)
    trainer.fit()
    finalize()


if __name__ == "__main__":
    main()
