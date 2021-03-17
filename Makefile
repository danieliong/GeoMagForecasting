.PHONY: sync_multirun

REMOTE = daniong@greatlakes-xfer.arc-ts.umich.edu
REMOTE_PROJECT_DIR = /home/daniong/geomag-forecasting
LOCAL_PROJECT_DIR = ~/geomag-forecasting

REMOTE_MULTIRUN_DIR = $(REMOTE_PROJECT_DIR)/multirun/
LOCAL_MULTIRUN_DIR = $(LOCAL_PROJECT_DIR)/multirun

REMOTE_OUTPUTS_DIR = $(REMOTE_PROJECT_DIR)/outputs/
LOCAL_OUTPUTS_DIR = $(LOCAL_PROJECT_DIR)/outputs/


sync_multirun:
	rsync -az --info=progress2 $(REMOTE):$(REMOTE_MULTIRUN_DIR) $(LOCAL_MULTIRUN_DIR)

sync_outputs:
	rsync -az --info=progress2 $(REMOTE):$(REMOTE_OUTPUTS_DIR) $(LOCAL_OUTPUTS_DIR)
