from abtest.loader import load


def test_vani_call_log_adapter():
    ds = load()
    assert not ds.synthetic
    c = ds.calls
    assert len(c) == 997 and c["caller_id"].nunique() == 825
    assert int(c["success"].sum()) == 517                    # BL Enriched + Approved + Generated
    assert not c.loc[c["success"], "connected"].eq(False).any()   # every lead came from a connected call
    assert c.loc[~c["connected"], "duration_s"].isna().all()      # no duration for calls that never connected
    assert c.loc[c["connected"], "duration_s"].notna().all()
