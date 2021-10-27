# Model directory
import diskcache

model_dir = '/host-dir/mip_models'

# Which device to use
device = 'cuda:0'

# Without setting to true, debugging will not work
debugging_enabled = False

# Caching
prefill_cache = False
cache_dir = "/tmp/cache"
cache = diskcache.FanoutCache(cache_dir,
                              size_limit=int((2 ** 30) * 100),  # 100Gb
                              cull_limit=0,
                              disk_pickle_protocol=-1)
