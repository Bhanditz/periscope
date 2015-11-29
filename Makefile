MMAP_FILES ?= ./tagged
DK_DATA ?= ./mp-dev_kit
MP_DATA ?= ./mp-data
OUTPUT_FILES ?= ./out
VENV = env/.built
PYTHON = env/bin/python3

RAW = $(MMAP_FILES)/full/train.labels.db \
      $(MMAP_FILES)/full/train.images.db \
      $(MMAP_FILES)/full/train.filenames.txt \
      #$(MMAP_FILES)/full/val.images.db \
      #$(MMAP_FILES)/full/test.images.db \

SRAW = $(MMAP_FILES)/small/train.labels.db \
       $(MMAP_FILES)/small/train.images.db \
       $(MMAP_FILES)/small/train.filenames.txt \
       #$(MMAP_FILES)/small/val.images.db \
       #$(MMAP_FILES)/small/test.images.db \

IMTGZ = $(MMAP_FILES)/data.tar.gz
IMDATA = $(MP_DATA)/images/train/y/yard/00001000.jpg

all: $(IMDATA) solve

$(VENV) env: env.sh
	sh env.sh
	touch $(VENV)

$(IMTGZ):
	mkdir -p $(MP_DATA)
	curl "http://6.869.csail.mit.edu/fa15/challenge/data.tar.gz" -o $@
	touch -d '2015-09-01' $@ # avoid rebuilds

.PRECIOUS: $(IMTGZ)

$(IMDATA): $(IMTGZ)
	tar mxvzf $< -C $(MP_DATA)

solve-small: $(VENV) $(SRAW) Makefile
	$(PYTHON) main.py -p plot.png -c network.mdl -e1 -b10 -s1 $(MMAP_FILES)/small

solve: $(VENV) $(RAW) Makefile
	$(PYTHON) main.py \
		-p plot-large.png \
		-c network-large.mdl \
		-e30 \
		$(MMAP_FILES)/full

analyze response-large.db confusion-large.db: $(VENV) $(RAW) Makefile
	$(PYTHON) main.py \
		-p plot-large.png \
		-c network-large.mdl \
		-e30 \
		-x confusion-large.db \
		-r response-large.db \
		$(MMAP_FILES)/full

view: $(VENV) response-large.db Makefile
	$(PYTHON) view.py \
		-r response-large.db \
                -t $(MMAP_FILES)/full \
                -d $(DK_DATA) \
                -o $(OUTPUT_FILES)

# these technically depend on $(PYTHON), but we don't want to add that
# dependency, because then we have to re-prepare if we ever change env.sh
$(SRAW): $(IMDATA) prepare.py
	mkdir -p $(MMAP_FILES)/small
	$(PYTHON) prepare.py -c10 -s200 $(MP_DATA)/images/ $(DK_DATA) $(MMAP_FILES)/small

$(RAW): $(IMDATA) prepare.py
	mkdir -p $(MMAP_FILES)/full
	$(PYTHON) prepare.py $(MP_DATA)/images/ $(DK_DATA) $(MMAP_FILES)/full

clean:
	rm -f $(RAW) $(SRAW) env
