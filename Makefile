
GREATLAKES_DIR = daniong@greatlakes-xfer.arc-ts.umich.edu:/home/daniong/geomag-forecasting

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
