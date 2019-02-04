.PHONY: compiler parser
compiler:
	pip3 install numpy
	pip3 install cma

parser: compiler
	pip3 install parse


