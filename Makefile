
GREATLAKES_DIR = daniong@greatlakes-xfer.arc-ts.umich.edu:/home/daniong/geomag-forecasting
PAPER_RESULTS = paper_results/

download_ace:
	python src/ace.py

sync_data:
	rsync -avz --exclude=ace_cdaweb/ \
		--exclude=ace/ \
		--exclude=supermag/ \
		data/ $(GREATLAKES_DIR)/data

compress_ace_cdaweb:
	./src/compress_ace_cdaweb.sh

uncompress_ace_cdaweb:
	./src/uncompress_ace_cdaweb.sh
retrieve_results_for_paper:
	python src/retrieve_results_from_mlflow.py \
		-m xgboost -l 60 -f ace_cdaweb -o $(PAPER_RESULTS)/xgboost_60_imf_only
	python src/retrieve_results_from_mlflow.py \
		-m xgboost -l 120 -f ace_cdaweb -o $(PAPER_RESULTS)/xgboost_120_imf_only
	python src/retrieve_results_from_mlflow.py \
		-m ebm -l 60 -f ace_cdaweb -o $(PAPER_RESULTS)/ebm_60_imf_only
	python src/retrieve_results_from_mlflow.py \
		-m ebm -l 120 -f ace_cdaweb -o $(PAPER_RESULTS)/ebm_120_imf_only
