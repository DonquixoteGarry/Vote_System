SHELL=cmd.exe
clean:
	del *.exe
	rmdir /S /Q data\cifar-10-batches-py
	cls

au:
	python autograd.py
cp:
	python cpuclassifer.py
	del cifar_net.pth
	rmdir /S /Q data\cifar-10-batches-py
nn:
	python nn.py
gp:
	cls
	python gpuclassifer.py
	rmdir /S /Q data\cifar-10-batches-py
g1:
	cls
	python grad1.py
g2:
	cls
	python grad2.py
mn:
	cls
	python mnist_nn.py
	cls