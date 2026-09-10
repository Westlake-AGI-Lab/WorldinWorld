"""Loading of the frozen LingBot-World-v2 14B causal-fast video model."""
import gc
import logging
import sys
import time

from .config import CKPT, LINGBOT_REPO, SINK_LAT, WINDOW_LAT


def load_pipe(ckpt_dir=None, device_id=0):
    sys.path.insert(0, LINGBOT_REPO)
    import wan
    from wan.configs import WAN_CONFIGS
    t0 = time.time()
    pipe = wan.WanI2VCausal(config=WAN_CONFIGS["i2v-A14B"], checkpoint_dir=ckpt_dir or CKPT, device_id=device_id,
                            rank=0, t5_cpu=True, local_attn_size=WINDOW_LAT, sink_size=SINK_LAT,
                            infer_mode="causal_fast")
    # prompts are pre-encoded by wiw.prep; the text encoder (~11 GB of host RAM) is not needed
    pipe.text_encoder = None
    gc.collect()
    logging.info(f"model loaded in {time.time() - t0:.0f}s")
    return pipe
