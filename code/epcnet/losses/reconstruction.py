from torch import Tensor, nn


class MaskedReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _selected(self, prediction: Tensor | None, target: Tensor, selected: Tensor) -> Tensor:
        if prediction is None or not selected.any():
            return target.new_zeros(())
        difference = prediction[selected] - target[selected]
        return difference.square().flatten(1).mean(dim=1).mean()

    def forward(
        self,
        electrochemical_prediction: Tensor | None,
        plasmonic_prediction: Tensor | None,
        electrochemical_target: Tensor,
        plasmonic_target: Tensor,
        modality_mask: Tensor,
    ) -> Tensor:
        electrochemical = self._selected(
            electrochemical_prediction,
            electrochemical_target,
            modality_mask[:, 0].bool(),
        )
        plasmonic = self._selected(
            plasmonic_prediction,
            plasmonic_target,
            modality_mask[:, 1].bool(),
        )
        return electrochemical + plasmonic
