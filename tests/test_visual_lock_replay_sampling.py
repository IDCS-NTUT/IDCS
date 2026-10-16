from types import SimpleNamespace as NS
import pytest
from jetson.visual_lock.frame_cache import bounded_catchup

def replay(times):
    return [(NS(frame_id=i,source_time_ns=t,tracker_session='A',source_clock_domain='pc_monotonic',
                width=1280,height=720,source_identity_verified=True),object()) for i,t in enumerate(times)]

def test_exact_seed_and_latest_image_survive_bounded_sampling():
    frames=replay([i*16_666_667 for i in range(16)])
    chosen=bounded_catchup(frames,100_000_000)
    assert chosen[0] is frames[0] and chosen[-1] is frames[-1]
    assert len(chosen)<len(frames)
    assert all(b[0].source_time_ns-a[0].source_time_ns <= 50_000_000 for a,b in zip(chosen,chosen[1:]))
    assert all(any(pair is original for original in frames) for pair in chosen)

def test_real_gap_is_preserved_for_fail_closed_tracker():
    frames=replay([0,20_000_000,140_000_000,160_000_000])
    assert bounded_catchup(frames,100_000_000)==frames

@pytest.mark.parametrize('field,value',[('tracker_session','B'),('source_clock_domain','other'),
    ('frame_id',0),('source_time_ns',0),('width',640),('source_identity_verified',False)])
def test_invalid_intermediate_frame_cannot_be_hidden(field,value):
    frames=replay([0,16_000_000,32_000_000,48_000_000])
    setattr(frames[1][0],field,value)
    assert bounded_catchup(frames,100_000_000)==frames

def test_slow_source_is_not_retimed_or_interpolated():
    frames=replay([0,60_000_000,120_000_000])
    assert bounded_catchup(frames,100_000_000)==frames
    assert bounded_catchup([],100_000_000)==[]
    assert bounded_catchup(frames[:1],100_000_000)==frames[:1]
    with pytest.raises(ValueError): bounded_catchup(frames,0)
