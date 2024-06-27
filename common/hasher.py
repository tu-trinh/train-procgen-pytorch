import torch
import hashlib
import numpy as np


class HashSet:
    def __init__(self):
        self.map = {}
        self.required_size = None
    
    def __len__(self):
        return len(self.map)
    
    def __str__(self):
        # string_keys = {}
        # for key in self.map:
        #     og_tensor = np.frombuffer(hashlib.md5(key.encode()).digest(), dtype = np.float32)
        #     string_keys[key] = str(og_tensor)
        # return "\n".join([f"{string_keys[key][:10]}: {vals}" for key, vals in self.map.items()])
        return "\n".join([f"{str(key).replace('(', '').replace(')', '')[:10]}: {vals}" for key, vals in self.map.items()])

    def tensor_to_hash(self, key_tensor):
        # tensor_bytes = key_tensor.float().cpu().numpy().tobytes()
        # return hashlib.md5(tensor_bytes).hexdigest()
        if isinstance(key_tensor, torch.Tensor):
            return self.tensor_to_hash(key_tensor.float().tolist())
        if isinstance(key_tensor, list):
            return tuple(self.tensor_to_hash(elem) for elem in key_tensor)
        return key_tensor

    def has_seen_key(self, key_tensor):
        if key_tensor.shape == self.required_size:
            hash_rep = self.tensor_to_hash(key_tensor)
            return hash_rep in self.map
        return False

    def has_seen_val(self, key_tensor, val_tensor):
        if key_tensor.shape == self.required_size:
            key_hash = self.tensor_to_hash(key_tensor)
            if key_hash in self.map:
                return val_tensor.float().cpu().item() in self.map[key_hash]
        return False

    def add_key(self, key_tensor):
        if self.required_size is None:
            self.required_size = key_tensor.shape
        assert key_tensor.shape == self.required_size
        key_hash = self.tensor_to_hash(key_tensor)
        if key_hash not in self.map:
            self.map[key_hash] = set()
    
    def add_val(self, key_tensor, val_tensor):
        if self.required_size is None:
            self.required_size = key_tensor.shape
        assert key_tensor.shape == self.required_size
        key_hash = self.tensor_to_hash(key_tensor)
        if key_hash not in self.map:
            self.map[key_hash] = set()
        self.map[key_hash].add(val_tensor.float().cpu().item())
    
    def get_vals(self, key_tensor):
        key_hash = self.tensor_to_hash(key_tensor)
        if key_hash in self.map:
            return self.map[key_hash]
        return None
    
    def reset(self, key_tensor):
        key_hash = self.tensor_to_hash(key_tensor)
        self.map[key_hash] = set()


if __name__ == "__main__":
    hash_set = HashSet()
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("A", a)
    b = torch.tensor([[1., 2., 3.0000000], [4.000, 5 - 0 + 3 - 3.0, 6.0000000000000000000000000000]])
    print("B", b)
    hash_set.add_key(a)
    assert hash_set.has_seen_key(b)
    hash_set.add_key(b)
    assert len(hash_set) == 1
    c = torch.tensor([[1, 2, 3], [4, 5, 6]]).reshape(3, 2)
    print("C", c)
    assert not hash_set.has_seen_key(c)
    d = torch.tensor([69])
    print("D", d)
    hash_set.add_val(a, d)
    try:
        hash_set.add_val(c, d)
    except AssertionError:
        assert True
    assert len(hash_set) == 1, len(hash_set)
    assert hash_set.has_seen_val(a, d)
    e = torch.tensor([420])
    print("E", e)
    hash_set.add_val(b, e)
    assert len(hash_set) == 1
    assert len(hash_set.get_vals(b)) == 2
    assert len(hash_set.get_vals(a)) == 2
    print(hash_set)
