from models.unet import UNet
from models.liif import LIIF

registry = {"unet": UNet, "liif": LIIF}
