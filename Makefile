
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
		-m xgboost -l 60 -f ace_cdaweb -o $(PAPER_RESULTS)/xgboost_60_imf_only \
		-s "param.booster='dart' and tags.features='imf_only'"
	python src/retrieve_results_from_mlflow.py \
		-m xgboost -l 60 -f ace_cdaweb -o $(PAPER_RESULTS)/xgboost_60_sw_imf \
		-s "param.booster='dart' and tags.features='sw_imf'"
	python src/retrieve_results_from_mlflow.py \
		-m xgboost -l 60 -f ace_cdaweb -o $(PAPER_RESULTS)/xgboost_60_sw_notemp \
		-s "param.booster='dart' and tags.features='sw_imf_notemp'"
	python src/retrieve_results_from_mlflow.py \
		-m xgboost -l 120 -f ace_cdaweb -o $(PAPER_RESULTS)/xgboost_120_imf_only
	python src/retrieve_results_from_mlflow.py \
		-m ebm -l 60 -f ace_cdaweb -s "tags.features='imf_only'"\
		-o $(PAPER_RESULTS)/ebm_60_imf_only
	python src/retrieve_results_from_mlflow.py \
		-m ebm -l 120 -f ace_cdaweb -s "tags.features='imf_only'" \
		-o $(PAPER_RESULTS)/ebm_120_imf_only
	python src/retrieve_results_from_mlflow.py \
		-m ebm -l 60 -f ace_cdaweb -s "tags.features='sw_imf'" \
		-o $(PAPER_RESULTS)/ebm_60_sw_imf
	python src/retrieve_results_from_mlflow.py \
		-m ebm -l 120 -f ace_cdaweb -s "tags.features='sw_imf'" \
		-o $(PAPER_RESULTS)/ebm_120_sw_imf
	python src/retrieve_results_from_mlflow.py \
		-m ebm -l 60 -f ace_cdaweb -s "tags.features='sw_imf2'" \
		-o $(PAPER_RESULTS)/ebm_60_sw_imf2

tune_xgboost_60_imf_only:
	python src/tune_hyperparams.py -m experiment_id='12' \
		features.name=ace_cdaweb split.method=storms_siciliano +data.start=1998-01-01

tune_xgboost_120_imf_only:
	python src/tune_hyperparams.py -m experiment_id='11' lagged_features.lead=120 \
		features.name=ace_cdaweb split.method=storms_siciliano +data.start=1998-01-01

train_xgboost_60_imf_only: tune_xgboost_60_imf_only
	python src/train.py experiment_id='5' tune.experiment_id='12' \
		features.name=ace_cdaweb split.method=storms_siciliano +data.start=1998-01-01

train_xgboost_120_imf_only: tune_xgboost_120_imf_only
	python src/train.py experiment_id='5' tune.experiment_id='11' lagged_features.lead=120 \
		features.name=ace_cdaweb split.method=storms_siciliano +data.start=1998-01-01
