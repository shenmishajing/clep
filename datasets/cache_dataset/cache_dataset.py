import copy
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import psutil
from mmengine.dataset import BaseDataset
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from tqdm import tqdm


class CacheDataset(BaseDataset):
    """
    Cache dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(lazy_init=True, *args, **kwargs)

    def parse_cache_info(self):
        for cache_name, cache_info in self.cache_info.items():
            cache_path = os.path.join(
                self.data_prefix["cache_path"],
                cache_name + "_cache",
            )
            if "path" in cache_info:
                cache_info["path"] = os.path.join(cache_path, cache_info["path"])
            else:
                cache_info["path"] = cache_path

            if "suffix" not in cache_info:
                cache_info["suffix"] = "pkl"

            if "func" not in cache_info:
                cache_info["func"] = self.cache_func

                if "kwargs" not in cache_info:
                    cache_info["kwargs"] = {}

                if "func" not in cache_info["kwargs"] and hasattr(
                    self, "cache_" + cache_name
                ):
                    cache_info["kwargs"]["func"] = getattr(self, "cache_" + cache_name)

            if "uncached" not in cache_info:
                cache_info["uncached"] = []

    @staticmethod
    def collate_fn(batch):
        collate_fn_map = copy.copy(default_collate_fn_map)
        collate_fn_map[list] = lambda batch, collate_fn_map: batch

        return collate(batch, collate_fn_map=collate_fn_map)

    def prepare_cache(self, name_list):
        for name in name_list:
            for info in self.cache_info.values():
                if not os.path.exists(
                    os.path.join(info["path"], name) + "." + info["suffix"]
                ):
                    info["uncached"].append(name)

        for info in self.cache_info.values():
            if info["uncached"]:
                info["func"](
                    info["uncached"],
                    cache_path=info["path"],
                    **info["kwargs"],
                )

                info["uncached"] = []

    def cache_func(self, data_list, cache_path, func, num_processes=None, **kwargs):
        os.makedirs(cache_path, exist_ok=True)

        if num_processes is None:
            num_processes = psutil.cpu_count(False)

        if num_processes:
            num_processes = min(num_processes, len(data_list))

            with ProcessPoolExecutor(num_processes) as executor:
                list(
                    tqdm(
                        executor.map(
                            partial(func, cache_path=cache_path, **kwargs),
                            data_list,
                            # chunksize=100,
                        ),
                        total=len(data_list),
                    )
                )
        else:
            for name in tqdm(data_list):
                func(name, cache_path, **kwargs)
