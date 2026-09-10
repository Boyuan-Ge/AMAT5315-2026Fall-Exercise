.PHONY: reproduce wolff analyze compare

ARTIFACTS ?= artifacts
ARTIFACTS_WOLFF ?= artifacts-wolff
BLOCKS ?= 50
SNAPSHOT_ARGS ?=
SWEEP_ARGS ?=
WOLFF_ARGS ?=

reproduce:
	cargo run --quiet --release -- snapshots --output "$(ARTIFACTS)/spins.jsonl" $(SNAPSHOT_ARGS)
	cargo run --quiet --release -- sweep --output "$(ARTIFACTS)" $(SWEEP_ARGS)
	cargo run --quiet --release -- analyze "$(ARTIFACTS)" --blocks $(BLOCKS)
	cargo run --quiet --release -- plot "$(ARTIFACTS)" --blocks $(BLOCKS)

wolff:
	cargo run --quiet --release -- sweep --wolff --output "$(ARTIFACTS_WOLFF)" $(WOLFF_ARGS)
	cargo run --quiet --release -- analyze "$(ARTIFACTS_WOLFF)" --blocks $(BLOCKS)
	cargo run --quiet --release -- plot "$(ARTIFACTS_WOLFF)" --blocks $(BLOCKS)

analyze:
	cargo run --quiet --release -- analyze "$(ARTIFACTS)" --blocks $(BLOCKS)

compare:
	cargo run --quiet --release -- compare-tau "$(ARTIFACTS)" "$(ARTIFACTS_WOLFF)" --l 64 --output tau-compare.png --blocks $(BLOCKS)
