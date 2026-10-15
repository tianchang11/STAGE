from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import functools
import numpy as np
import math
import torch.distributed as dist
import torch
import pickle


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks.
    Cached — creating a new group every epoch leaks process groups / can hang.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        print(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                dist.get_rank(), len(buffer) / (1024**3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert world_size >= 1, "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size.item() != max_size:
        padding = torch.zeros((max_size - local_size.item(),), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if dist.get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


class RandomIdentitySampler_DDP(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Builds *global* batches of size ``batch_size`` (same structure as single-GPU
    ``RandomIdentitySampler``), then shards each global batch across ranks so
    that after feature all-gather the union matches single-GPU training.

    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): global number of examples in a batch (IMS_PER_BATCH).
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.world_size = dist.get_world_size()
        self.num_instances = num_instances
        self.mini_batch_size = self.batch_size // self.world_size
        if self.batch_size % self.world_size != 0:
            raise ValueError(
                f"IMS_PER_BATCH ({batch_size}) must be divisible by world_size ({self.world_size})"
            )
        if self.mini_batch_size % self.num_instances != 0:
            raise ValueError(
                f"per-GPU batch ({self.mini_batch_size}) must be divisible by "
                f"NUM_INSTANCE ({self.num_instances})"
            )
        # Global identity count per step — same as single-GPU RandomIdentitySampler
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)

        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch (per rank)
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        self.rank = dist.get_rank()
        self.length //= self.world_size
        self.epoch = 0
        self._seed = 0

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic, non-overlapping shuffle across ranks."""
        self.epoch = int(epoch)

    def __iter__(self):
        # Prefer epoch-based shared seed (stable + set_epoch friendly);
        # still all-gather once so every rank agrees even if epoch differs.
        seed = shared_random_seed() ^ (self.epoch + 1) * 0x9E3779B9
        self._seed = int(seed & 0x7FFFFFFF)
        rng = np.random.RandomState(self._seed)
        final_idxs = self.sample_list(rng)
        length = int(math.ceil(len(final_idxs) * 1.0 / self.world_size))
        final_idxs = self.__fetch_current_node_idxs(final_idxs, length)
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __fetch_current_node_idxs(self, final_idxs, length):
        """
        Shard global index list into per-rank mini-batches.

        Global layout: [GB0 (batch_size), GB1, ...]
        Each global batch is split into world_size contiguous mini-batches of
        size mini_batch_size (identities already interleaved as pid blocks of K).
        """
        total_num = len(final_idxs)
        # Number of complete global batches
        n_global = total_num // self.batch_size
        index_target = []
        for g in range(n_global):
            base = g * self.batch_size
            start = base + self.rank * self.mini_batch_size
            end = start + self.mini_batch_size
            index_target.extend(range(start, min(end, total_num)))
        index_target_npy = np.array(index_target, dtype=np.int64)
        if index_target_npy.size == 0:
            return []
        final_idxs = list(np.array(final_idxs)[index_target_npy])
        return final_idxs

    def sample_list(self, rng=None):
        """
        Build a list of indices as concatenated *global* batches — identical
        structure to single-GPU RandomIdentitySampler (N pids x K instances).
        """
        if rng is None:
            rng = np.random.RandomState(self._seed)

        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = rng.choice(idxs, size=self.num_instances, replace=True).tolist()
            else:
                idxs = list(idxs)
            rng.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        batch_indices = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = rng.choice(
                avai_pids, self.num_pids_per_batch, replace=False
            ).tolist()
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                batch_indices.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return batch_indices

    def __len__(self):
        return self.length
