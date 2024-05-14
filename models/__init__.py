from models.gwcnet import GwcNet_G, GwcNet_GC,PSM_HCR
from models.loss import model_loss

__models__ = {
    "gwcnet-g": GwcNet_G,
    "gwcnet-gc": GwcNet_GC,
    "psm-hcr": PSM_HCR, #7.7M
}
