import shutil
from pathlib import Path

import clip
import cv2
import faiss
import torch
from einops import rearrange
from loguru import logger
from tqdm import tqdm
import numpy as np
import h5py
import atexit
from utils.memoize import MemoizeNumpy

from utils.train_helper import preprocess_for_CLIP

@MemoizeNumpy
def get_image_vector(img_path: Path, clip_model):
    img = cv2.imread(str(img_path))[:, :, ::-1] / 255.0
    img = rearrange(img, "h w c -> 1 c h w")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img = preprocess_for_CLIP(torch.tensor(img).to(device))

    img_features = clip_model.encode_image(img).detach().cpu()

    # Normalize
    faiss.normalize_L2(img_features.numpy().astype("float32"))

    img_features = img_features.squeeze(0)

    return img_features.numpy()


def get_img_feature_ll(rgb_file_ll):
    pbar = tqdm(total=len(rgb_file_ll))
    img_feature_ll = []
    for rgb_file in rgb_file_ll:
        pbar.set_description(f"File {rgb_file}")
        img_feature_ll.append(get_image_vector(rgb_file, clip_model))
        pbar.update(1)

    img_feature_ll = np.stack(img_feature_ll, axis=0).astype("float32")
    return img_feature_ll


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# CLIP
logger.info("Loading CLIP model")
clip_model, _ = clip.load("ViT-B/32", device=device)

dataset_folder = Path("data/celeba-hq")
rgb_path = dataset_folder / "rgb"

hdf5_file = dataset_folder / "clip_embeddings.hdf5"
hdf5_key = "clip_embeddings"

caption = "This is president bill clinton"
text = clip.tokenize([caption]).to(device)
text_features = clip_model.encode_text(text).cpu().detach().numpy().astype("float32")
faiss.normalize_L2(text_features)

# Glob files
extension = "jpg"
logger.info(f"Globbing {extension} images")
rgb_file_ll = list(rgb_path.glob(f"*.{extension}"))

n_dim = text_features.shape[1]

# FAISS CPU
index = faiss.IndexFlatIP(n_dim)

# Get image features
dump_features = True
if hdf5_file.exists():
    logger.info(f"Found embedding hdf5 file {hdf5_file}")
    hdf5_object = h5py.File(f"{hdf5_file}", "a")
    dset = hdf5_object.get(hdf5_key)

    if dset.shape[0] == len(rgb_file_ll):
        logger.info(f"Dataset has {len(rgb_file_ll)} embeddings, matching.")
        dump_features = False

        # Create empty numpy array
        img_feature_ll = np.zeros(shape=dset.shape, dtype=dset.dtype)
        dset.read_direct(img_feature_ll)
    else:
        logger.info(f"Dataset has only {dset.shape} embeddings, not matching {len(rgb_file_ll)}")

    hdf5_object.close()

if dump_features:
    img_feature_ll = get_img_feature_ll(rgb_file_ll)

    hdf5_object = h5py.File(f"{hdf5_file}", "a")
    hdf5_object.create_dataset(hdf5_key, data=img_feature_ll, compression="lzf")
    atexit.register(hdf5_object.close)

# Add to index
logger.info("Building faiss index")
index.add(img_feature_ll)

# Search!
n_neighbours = 10
logger.info("Searching faiss index")
distance_ll, index_ll = index.search(text_features, n_neighbours)

matched_rgb_files = np.array(rgb_file_ll)[index_ll.squeeze(0)]

out_dir = rgb_path.parent / "similarities"
out_dir.mkdir(exist_ok=True, parents=True)

for rgb_file, distance in zip(matched_rgb_files, distance_ll.squeeze(0)):
    logger.info(f"Distance for {rgb_file} is {distance}")
    shutil.copy(rgb_file, out_dir / rgb_file.name)
