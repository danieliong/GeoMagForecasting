.PHONY: sync_multirun_to_local, sync_multirun_to_remote, sync_outputs_to_local,
		sync_outputs_to_remote, install_src, data_greatlakes

REMOTE = daniong@greatlakes-xfer.arc-ts.umich.edu
REMOTE_PROJECT_DIR = /home/daniong/geomag-forecasting
LOCAL_PROJECT_DIR = ~/geomag-forecasting

REMOTE_MULTIRUN_DIR = $(REMOTE_PROJECT_DIR)/multirun/
LOCAL_MULTIRUN_DIR = $(LOCAL_PROJECT_DIR)/multirun

REMOTE_OUTPUTS_DIR = $(REMOTE_PROJECT_DIR)/outputs/
LOCAL_OUTPUTS_DIR = $(LOCAL_PROJECT_DIR)/outputs/

sync_multirun_to_local:
	rsync -az --info=progress2 $(REMOTE):$(REMOTE_MULTIRUN_DIR) $(LOCAL_MULTIRUN_DIR)

sync_multirun_to_remote:
	rsync -az --info=progress2 $(LOCAL_MULTIRUN_DIR) $(REMOTE):$(REMOTE_MULTIRUN_DIR)

sync_outputs_to_local:
	rsync -az --info=progress2 $(REMOTE):$(REMOTE_OUTPUTS_DIR) $(LOCAL_OUTPUTS_DIR)

sync_outputs_to_remote:
	rsync -az --info=progress2  $(LOCAL_OUTPUTS_DIR) $(REMOTE):$(REMOTE_OUTPUTS_DIR)

install_src:
	pip install -e .

data_greatlakes:
	python src/process_data.py \
		target.loading.data_dir=/home/daniong/scratch/supermagstations/data \
		hydra.verbose=__main__
