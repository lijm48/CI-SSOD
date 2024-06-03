pre:
	python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
	mkdir -p thirdparty
	# git clone https://github.com/open-mmlab/mmdetection.git thirdparty/mmdetection
	wget https://github.com/open-mmlab/mmdetection/archive/refs/tags/v2.16.0.tar.gz 
	tar -zxvf v2.16.0.tar.gz 
	mv mmdetection-2.16.0 thirdparty/mmdetection
	cd thirdparty/mmdetection && python -m pip install -e .
install:
	make pre
	python -m pip install -e .
clean:
	rm -rf thirdparty
	rm -r ssod.egg-info