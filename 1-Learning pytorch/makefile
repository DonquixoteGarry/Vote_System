SHELL=cmd.exe
clean:
	del *.exe
	rmdir /S /Q data\cifar-10-batches-py runs
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
tu:
	cls
	echo .. open following URL and read it with notebook
	echo https://pytorch.apachecn.org/#/docs/1.7/16
	jupyter notebook
te:
	cls
	rmdir /S /Q runs
	echo tensorboard --logdir=runs
	echo https://pytorch.apachecn.org/#/docs/1.7/17
	jupyter notebook