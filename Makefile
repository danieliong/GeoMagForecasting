download_ace:
# FIXME: This didn't work
	wget -r -nH -nc -P data/ --cut-dirs=2 --no-parent --accept *_ace_{swepam,mag}_1m.txt https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/ 

download_ace_positions:
	wget -r -nH -P data/ace/positions --cut-dirs=4 --no-parent --accept *_ace_loc_1h.txt https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/
