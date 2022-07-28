from models.cfnet import CFNet
from models.gwcnet import GwcNet
from models.loss import model_loss

__models__ = {
    "cfnet": CFNet,
    "GwcNet":GwcNet
}
