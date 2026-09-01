.PHONY: test toy pals lm-pilot bundle

test:
	PYTHONPATH=src pytest -q

toy:
	python scripts/run_toy.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11

pals:
	python scripts/generate_pals.py --seed 20260901

lm-pilot:
	python scripts/run_lm_pals.py --method promotion --seed 20260901 --eval-cap 4

bundle:
	git archive --format=zip --output=../state-promotion-repo.zip HEAD
