# Makefile for gtrack examples

DATA_URL := https://data.gadopt.org/demos/Matthews_et_al_410_0.tar.gz
DATA_TARBALL := Matthews_et_al_410_0.tar.gz
DATA_DIR := Matthews_et_al_410_0
DEMOS_DIR := ../docs/demos

# Data from Hoggard et al 2022
# OSF data (lithospheric thickness maps)
OSF_PROJECT := twksd
OSF_FOLDER_ID := 5ecefa09aeeb6d0286087b87
OSF_DATA_DIR := lithospheric_thickness_maps

# Interpolated lithospheric thickness data (sphere mesh)
LITHO_NC := $(OSF_DATA_DIR)/global/SL2013sv/SL2013sv.nc
LITHO_H5 := output/lithospheric_thickness_mesh.h5
LITHO_NPOINTS := 40000

# Python scripts to convert to notebooks
EXAMPLE_SCRIPTS := seafloor_age_simple.py seafloor_age_stepwise.py continental_rotation.py
NOTEBOOKS := $(patsubst %.py,$(DEMOS_DIR)/%.ipynb,$(EXAMPLE_SCRIPTS))

.PHONY: all clean data run notebooks clean-output clean-notebooks osf-data clean-osf litho-data clean-litho

all: data run

# Download and extract plate model data
data: $(DATA_DIR)

$(DATA_DIR): $(DATA_TARBALL)
	@echo "Extracting plate model data..."
	tar -xzf $(DATA_TARBALL)
	@echo "Done."

$(DATA_TARBALL):
	@echo "Downloading plate model data..."
	curl -L -o $(DATA_TARBALL) $(DATA_URL)
	@echo "Download complete."

# Download lithospheric thickness maps from OSF
osf-data: $(OSF_DATA_DIR)

$(OSF_DATA_DIR):
	@echo "Downloading lithospheric thickness maps from OSF..."
	python download_osf_folder.py -p $(OSF_PROJECT) -f $(OSF_FOLDER_ID) -o $(OSF_DATA_DIR)
	@echo "OSF data download complete."

# Interpolate lithospheric thickness to sphere mesh
# This creates uniformly-spaced points suitable for numerical simulations
litho-data: $(LITHO_H5)

$(LITHO_H5): $(LITHO_NC)
	@echo "Interpolating lithospheric thickness to sphere mesh..."
	@mkdir -p output
	python interpolate_to_mesh.py -i $(LITHO_NC) -o $(LITHO_H5) -n $(LITHO_NPOINTS)
	@echo "Interpolation complete: $(LITHO_H5)"

$(LITHO_NC): $(OSF_DATA_DIR)
	@test -f $(LITHO_NC) || (echo "Error: $(LITHO_NC) not found. Run 'make osf-data' first." && exit 1)

# Run examples (without generating notebooks)
run: data
	@echo "Running seafloor_age_simple.py..."
	python seafloor_age_simple.py
	@echo "Running seafloor_age_stepwise.py..."
	python seafloor_age_stepwise.py
	@echo "Examples complete."

# Generate Jupyter notebooks from Python scripts
notebooks: data $(NOTEBOOKS)
	@echo "All notebooks generated in $(DEMOS_DIR)/"

$(DEMOS_DIR)/%.ipynb: %.py
	@mkdir -p $(DEMOS_DIR)
	@echo "Converting $< to notebook and executing..."
	jupytext --to ipynb --execute $<
	mv $*.ipynb $@
	@echo "Generated $@"

# Clean downloaded data
clean:
	rm -rf $(DATA_DIR)
	rm -f $(DATA_TARBALL)

# Clean output files
clean-output:
	rm -rf output/
	rm -f *.png

# Clean generated notebooks
clean-notebooks:
	rm -rf $(DEMOS_DIR)

# Clean OSF data
clean-osf:
	rm -rf $(OSF_DATA_DIR)/

# Clean interpolated lithospheric data
clean-litho:
	rm -f $(LITHO_H5)
