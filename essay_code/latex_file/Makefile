SHELL=cmd.exe
show:
	python show.py
compile:
	xelatex main
	biber main
	xelatex main
	xelatex main
	del *.aux *.out *.blg *.toc *.bbl *.bcf
clean-all:
	del *.aux *.out *.blg *.toc *.bbl *.bcf *.synctex.gz 
	del *.vrb *.log *.nav *.snm *.xml
clean2:	
	del main.pdf main.run.xml main.log
zip:
	zip folder/my_essay.zip *.tex *.bbx *.bib *.cbx *.sty fonts/* figures/*
main:
	del *.aux *.out *.blg *.toc *.bbl *.bcf *.synctex.gz
	del *.vrb *.log *.nav *.snm *.xml
	del main.pdf main.run.xml main.log
	xelatex main
	biber main
	xelatex main
	xelatex main
	del *.aux *.out *.blg *.toc *.bbl *.bcf *.synctex.gz
	del *.vrb *.log *.nav *.snm *.xml
	cls
