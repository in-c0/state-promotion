.PHONY: test toy pals lm-pilot lm-pilot-revision bundle

test:
	PYTHONPATH=src pytest -q

toy:
	python scripts/run_toy.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11

pals:
	python scripts/generate_pals.py --seed 20260901

lm-pilot:
	python scripts/run_exp001_pilot.py --seed 20260901 --stream retention --eval-cap 4

lm-pilot-revision:
	python scripts/run_exp001_pilot.py --seed 20260901 --stream revision --eval-cap 4

bundle:
	git archive --format=zip --output=../state-promotion-repo.zip HEAD
