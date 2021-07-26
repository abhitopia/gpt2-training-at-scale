import hashlib
from pathlib import Path
from tqdm import tqdm

HASH_FUNCS = {
    'md5': hashlib.md5,
    'sha1': hashlib.sha1,
    'sha256': hashlib.sha256,
    'sha512': hashlib.sha512
}


def hash_files(paths, hash_func='md5'):
    hashfunc = HASH_FUNCS.get(hash_func)

    hash_vals = []
    pbar = tqdm(paths, desc='Hashing data files:')
    for path in pbar:
        pbar.set_postfix({'File': path.name})
        hash_vals.append(_filehash(str(Path(path).absolute()), hashfunc))

    return _reduce_hash(hash_vals, hashfunc)


def _filehash(filepath, hashfunc):
    hasher = hashfunc()
    size = str(Path(filepath).stat().st_size)
    name = str(Path(filepath).absolute())
    hasher.update(name.encode('utf-8'))
    hasher.update(size.encode('utf-8'))
    return hasher.hexdigest()

# def _filehash(filepath, hashfunc):
#     hasher = hashfunc()
#     blocksize = 64 * 1024
#
#     if not os.path.exists(filepath):
#         return hasher.hexdigest()
#
#     with open(filepath, 'rb') as fp:
#         while True:
#             data = fp.read(blocksize)
#             if not data:
#                 break
#             hasher.update(data)
#     return hasher.hexdigest()


def _reduce_hash(hashlist, hashfunc):
    hasher = hashfunc()
    for hashvalue in sorted(hashlist):
        hasher.update(hashvalue.encode('utf-8'))
    return hasher.hexdigest()


def hash_string(text):
    return int(hashlib.md5(text.encode('utf8')).hexdigest()[:8], 16)
