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
	cls
	python gpurun.py
	rmdir /S /Q data\cifar-10-batches-py
g1:
	cls
	python grad1.py
clean:
	del *.exe
	rmdir /S /Q data\cifar-10-batches-py
	cls
