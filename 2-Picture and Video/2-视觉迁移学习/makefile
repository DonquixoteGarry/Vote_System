SHELL:cmd.exe
tr: pre2 pre1 pre3
pre1:
	echo delete_original_folder,unzip_zipball,start_jupyter_notebook,
pre2:
	unzip hymenoptera_data.zip -d .\data

pre3:
	jupyter notebook
clean:
	rmdir /S /Q .\data 
1:
	vim makefile
