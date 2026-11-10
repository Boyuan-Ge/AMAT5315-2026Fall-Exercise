"""Read-only audit of the committed figures, numerical results and saved plans."""
import re
from xml.etree import ElementTree
from PIL import Image
import numpy as np
from evidence import ROOT, OUT, read, recount


def main():
    names = ["graphs.svg", "posteriors.png", "joint.png", "orders.png", "asia-bp.png",
             "promedus-cost.png", "promedus-bp.png"]
    embeds = re.findall(r"!\[[^\]]*\]\(evidence/([^)]*)\)", (ROOT/"README.md").read_text())
    assert sorted(embeds) == sorted(names)
    for name in names:
        if name.endswith(".svg"):
            root = ElementTree.parse(OUT/name).getroot()
            assert root.tag.endswith("}svg") and "Open T" in "".join(root.itertext())
        else:
            with Image.open(OUT/name) as image:
                assert min(image.size) >= 300
                image.verify()
    a = read("evidence/asia.json")
    for name, mass in zip(("prior", "AD", "ADX"), (1., .004501375, .00098822675)):
        exact=a[name]["exact"]
        assert abs(exact["tenferro"]["evidence_mass"]/mass-1)<1e-10
        assert max(exact["marginal_max_absolute_error"],exact["evidence_relative_error"],exact["normalization_error"])<1e-10
        check=exact["numpy_check"]
        assert max(check["matrix_max_error"],check["trace_error"],check["ising_error"])<1e-12
        assert a[name]["bp"]["converged"]
    assert a["ADX"]["trace"][-1][2] > .01
    o=read("evidence/orders.json")
    for name in ("random","greedy"):
        assert read(f"evidence/ad-{name}-plan.json")==o["plans"][name]
        assert recount(o["plans"][name], [2]*8, o["complexities"][name])==o["recounted"][name]
    assert o["exhaustive_minimum_work"]=={"prior":44,"AD":50,"ADX":52}
    b=read("evidence/promedus.json")
    for name in ("greedy","treesa"):
        plan=read(f"evidence/promedus-{name}-plan.json")
        cost=read(f"evidence/promedus-{name}-complexity.json")
        assert recount(plan,[2]*414,cost)==b[name]["scalar_cost"]
        actual=b[name]["run"]
        assert abs(actual["log10_partition"]-b["published_PR"])<1e-4
        for key,marginal in actual["marginals"].items():
            np.testing.assert_allclose(marginal,b["published_MAR"][key],atol=1e-5,rtol=0)
    assert abs(b["greedy"]["run"]["log10_partition"]-b["treesa"]["run"]["log10_partition"])<1e-10
    for key, marginal in b["greedy"]["run"]["marginals"].items():
        np.testing.assert_allclose(marginal,b["treesa"]["run"]["marginals"][key],atol=1e-10,rtol=0)
    assert b["bp"]["sweeps"]==1000 and b["bp"]["residual"]>1e-10
    assert not b["bp"]["converged"] and b["bp_trace"][-1][2]>.38
    assert read("evidence/environment.json")["rustc"].startswith("rustc 1.96.0 ")
    print("Committed evidence verified: seven inline figures, ASIA checks, four saved plan recounts, and Promedus PR/MAR/BP.")


if __name__ == "__main__":
    main()
