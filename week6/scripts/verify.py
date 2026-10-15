"""Acceptance checks against the learning sheet and saved experimental records."""
import re
import tomllib
from common import *
from hopfield import summarize_random
from rbm import summarize_sampling


def main():
    summarize_random()
    summarize_sampling()
    config=tomllib.loads((ROOT/"rbm.toml").read_text())
    assert config["training"]=={"images":60000,"epochs":20,"batch_size":100,"shuffle_each_epoch":True,
        "optimizer":"SGD","learning_rate":.1,"momentum":0.,"weight_decay":0.,"precision":"float32","seed":6602}
    train=json.loads((EVIDENCE/"rbm-training.json").read_text())
    assert train["settings"]["hidden_units"]==128
    a,b=(train["models"][k] for k in ("cd1","cd10"))
    assert a["initial_hash"]==b["initial_hash"] and a["shuffle_hashes"]==b["shuffle_hashes"]
    assert len(a["shuffle_hashes"])==20
    for model in (a,b):
        h=model["history"]
        assert [r["epoch"] for r in h]==list(range(21))
        assert h[-1]["reconstruction_error"]<.5*h[0]["reconstruction_error"]
    x,y,tx,ty=load_mnist()
    truth=(tx[:64].reshape(-1,784)>=128).astype(np.float32)
    starts=np.load(ART/"rbm/matched-starts.npz")
    wrong=starts["wrong_upper_indices"]
    assert np.all(ty[:64]!=ty[wrong])
    previous_generation=None
    for k in (1,10):
        frames=np.load(ART/"rbm"/f"samples-cd{k}.npz")
        assert frames["generation"].shape==(5,64,784)
        assert np.array_equal(frames["generation"][0],starts["generation"])
        if previous_generation is not None:assert np.array_equal(previous_generation,frames["generation"][0])
        previous_generation=frames["generation"][0]
        for condition,indices in (("true",np.arange(64)),("wrong",wrong)):
            f=frames[f"completion_{condition}"]
            assert f.shape==(5,64,8,784)
            target=np.broadcast_to(truth[indices][None,:,None,:392],f[...,:392].shape)
            assert np.array_equal(f[...,:392],target)
            assert np.array_equal(f[0,:,:,392:],starts["completion"][:,:,392:])
            assert set(np.unique(f))<={0.,1.}
    sampling=json.loads((EVIDENCE/"rbm-sampling.json").read_text())
    for model in sampling["models"].values():assert model["true_wrong_gap_1000"]>=.1
    random_runs={}
    for n in (100,200,400,800):
        raw=json.loads((ART/"hopfield"/f"random-N{n}.json").read_text())
        assert raw["pattern_sets"]==20 and len(raw["rows"])==21
        seeds=[]
        for tick,row in zip(range(5,26),raw["rows"]):
            assert row["p"]==round(n*tick/100) and len(row["trials"])==20
            for trial in row["trials"]:
                seeds.append(trial["seed"])
                assert len(trial["overlaps"])==min(10,row["p"])
                assert all(s>0 for s in trial["sweeps"])
                assert all(-1<=v<=1 for v in trial["overlaps"])
        assert len(set(seeds))==420
        random_runs[n]={"seconds":raw["seconds"],"pattern_sets":420,
                        "recall_starts":sum(min(10,r["p"])*20 for r in raw["rows"])}
    crossings=json.loads((EVIDENCE/"capacity-crossings.json").read_text())
    assert crossings["decreases_with_N"] and crossings["within_reference_tolerance"]
    modern=json.loads((EVIDENCE/"modern-metrics.json").read_text())
    assert len(modern["rows"])==42 and modern["attention_max_error"]<1e-5
    for r in modern["rows"]:
        assert r["classical_success"]==0
        if r["beta"]==.3:assert r["modern_success"]>=.9
    figures=sorted(EVIDENCE.glob("*.png"))
    assert len(figures)==15,len(figures)
    if (ROOT/"README.md").exists():
        readme=(ROOT/"README.md").read_text()
        for p in figures:
            assert f"({p.relative_to(ROOT).as_posix()})" in readme,p
    assert all(p.stat().st_size<5_000_000 for p in EVIDENCE.iterdir() if p.is_file())
    report={"all_required_numerical_checks_passed":True,"dataset_original_splits":True,
            "training_images":60000,"epochs":20,"hidden_units":128,"matched_initialization":True,
            "matched_training_order":True,"reconstruction_less_than_half_initial":True,
            "matched_fresh_generation_starts":True,"completion_upper_half_exactly_clamped":True,
            "completion_correct_minus_wrong_at_1000":{k:v["true_wrong_gap_1000"] for k,v in sampling["models"].items()},
            "random_hopfield_runs":random_runs,"capacity_crossings":crossings["crossings"],
            "modern_attention_max_error":modern["attention_max_error"],
            "modern_beta_point3_min_success":min(r["modern_success"] for r in modern["rows"] if r["beta"]==.3),
            "classical_digit_success":0,"embedded_result_figures":len(figures),
            "raw_runs_and_downloads":"ignored; retained locally under artifacts/ and inputs/",
            "environment":environment()}
    write_json(EVIDENCE/"verification.json",report)
    write_json(ART/"environment.json",environment())
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
