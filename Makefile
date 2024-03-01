FIGURES:=Fig6 Fig9 Fig11 Fig10 Fig4 Fig5 Fig8 Fig7
FIGURE_DIR:=figures/
TIFF_FIGURES:=$(addprefix $(FIGURE_DIR),$(FIGURES:=.tiff))
SVG_FIGURES:=$(addprefix $(FIGURE_DIR),$(FIGURES:=.svg))
PNG_FIGURES:=$(addprefix $(FIGURE_DIR),$(FIGURES:=.png))

all: $(SVG_FIGURES) $(TIFF_FIGURES) $(PNG_FIGURES)

figures.zip: $(TIFF_FIGURES)
	zip figures $(TIFF_FIGURES)

.PHONY:
png: $(PNG_FIGURES)

.PHONY:
svg: $(SVG_FIGURES)

.PHONY: clean_tiff
clean_tiff:
	find $(FIGURE_DIR) -type f -name "*.tiff" -delete

%.tiff: %.png
	magick mogrify -compress lzw -format tiff $<

%.png: %.svg
	inkscape -o $@ -d 600 $<

figures/Fig6.svg: code/TFR_k+_k-_M.py
	cd figures && python -m TFR_k+_k-_M

figures/Fig9.svg: code/five_panel.py
	cd figures && python -m five_panel

figures/Fig11.svg: code/plots_2^11_800.py
	cd figures && python -m plots_2^11_800

figures/Fig10.svg: code/plot_together.py code/bifurcation_u.py code/eigenvalues_and_SS_u.py
	cd figures && python -m plot_together

figures/Fig4.svg: code/system_stepfunct_volumeterm.py
	cd figures && python -m system_stepfunct_volumeterm

figures/Fig5.svg: code/TFR_k1_k-1_T_a.py
	cd figures && python -m TFR_k1_k-1_T_a

figures/Fig8.svg: code/plots_2^11_800_4&5params.py
	cd figures && python -m "plots_2^11_800_4&5params"

figures/Fig7.svg: code/plots_2^11_800_4&5params.py
	cd figures && python -m "plots_2^11_800_4&5params"
