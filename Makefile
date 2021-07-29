GDOWN := $(shell command -v gdown 2> /dev/null)
DOWNLOADED := $(shell ls ./checkpoints/mangainpaintor 2> /dev/null)

install-gdown:
ifndef GDOWN
	pip install gdown
endif

download-models: install-gdown
ifndef DOWNLOADED
	gdown --id 1YeVwaNfchLhy3lAA7jOLBP-W23onjy8S --output mangainpainter.zip
	gdown --id 1QaXqR4KWl_lxntSy32QpQpXb-1-EP7_L --output ScreenVAE.zip
	unzip mangainpainter.zip -d ./checkpoints
	unzip ScreenVAE.zip -d ./checkpoints
	rm mangainpainter.zip ScreenVAE.zip
endif

test:
	python test.py --checkpoints ./checkpoints/mangainpaintor/ \
    	--input examples/test/imgs/ \
    	--mask examples/test/masks/ \
    	--line examples/test/lines/ \
    	--output examples/test/results/