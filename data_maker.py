
from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn

import argparse
from io import BytesIO
import multiprocessing
from functools import partial


def RW(img_file, sizes, resample):
    i, file = img_file
    img = Image.open(file)
    img = img.convert("RGB")
    out = RNM(img, sizes=sizes, resample=resample)

    return i, out

def RNC(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val

def prepare(
    env, dataset, n_worker, sizes=(32, 128, 256, 512, 1024), resample=Image.LANCZOS
):
    resize_fn = partial(RW, sizes=sizes, resample=resample)

    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))

def RNM(
    img, sizes=(32, 128, 256, 512, 1024), resample=Image.LANCZOS, quality=100
):
    imgs = []

    for size in sizes:
        imgs.append(RNC(img, size, resample, quality))

    return imgs

def data_generator(size, path, num_of_workers, out, resample=lanczos):

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[resample]

    sizes = [int(s.strip()) for s in size.split(",")]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(path)

    with lmdb.open(out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, num_of_workers, sizes=sizes, resample=resample)
