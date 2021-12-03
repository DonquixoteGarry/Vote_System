SHELL=cmd.exe
au:
	python autograd.py
cl:
	python classifer.py
	del cifar_net.pth
	rmdir /S /Q data\cifar-10-batches-py
nn:
	python nn.py
gp:
	python gpurun.py
	rmdir /S /Q data\cifar-10-batches-py
clean:
	del *.exe
	rmdir /S /Q data\cifar-10-batches-py
	cls
