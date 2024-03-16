CXX = g++
OPENCV_INCLUDE = `pkg-config opencv4 --cflags`
OPENCV_LIB = `pkg-config opencv4 --libs`
WEBOTS_HOME = $(HOME)/webots
WEBOTS_LIB_DIR = $(WEBOTS_HOME)/lib/controller
WEBOTS_INCLUDE_DIR = $(WEBOTS_HOME)/include/controller/c
INCLUDES = -I$(WEBOTS_INCLUDE_DIR) -I/usr/local/include -I/usr/include/lua5.1 -I. -I/usr/include/python3.6m $(OPENCV_INCLUDE)

main: Swig.o Code_wrap.o luaModel.o luatables.o ukfmodel.o ActiveVision.o BallModel.o Transform.o MatrixTransform.o HeadTransform.o
	$(CXX) -shared Code_wrap.o luaModel.o luatables.o ukfmodel.o ActiveVision.o BallModel.o Transform.o MatrixTransform.o HeadTransform.o -o _Code.so -lpthread -lm -larmadillo -llua5.1 $(OPENCV_LIB) -L$(WEBOTS_LIB_DIR) -lController
	cp Code.py _Code.so ../DQNController/

Swig.o: Code.i
	swig -python -c++ Code.i

Code_wrap.o: Code_wrap.cxx
	$(CXX) -c -fpic Code_wrap.cxx -O2 -fpic $(INCLUDES) -o Code_wrap.o

luaModel.o: luaModel.cpp luaModel.h
	$(CXX) -c -fpic luaModel.cpp -O2 -fpic $(INCLUDES) -o luaModel.o

luatables.o: luatables.cpp luatables.h
	$(CXX) -c -fpic luatables.cpp -O2 -fpic $(INCLUDES) -o luatables.o

ukfmodel.o: ukfmodel.cpp ukfmodel.h
	$(CXX) -c -fpic ukfmodel.cpp -O2 -fpic $(INCLUDES) -o ukfmodel.o

ActiveVision.o: ActiveVision.cpp ActiveVision.h
	$(CXX) -c -fpic ActiveVision.cpp -O2 -fpic $(INCLUDES) -o ActiveVision.o

BallModel.o: BallModel.cpp BallModel.h
	$(CXX) -c -fpic BallModel.cpp -O2 -fpic $(INCLUDES) -o BallModel.o

Transform.o: Transform.cpp Transform.h
	$(CXX) -c -fpic Transform.cpp -O2 -fpic $(INCLUDES) -o Transform.o

MatrixTransform.o: MatrixTransform.cpp MatrixTransform.h
	$(CXX) -c -fpic MatrixTransform.cpp -O2 -fpic $(INCLUDES) -o MatrixTransform.o

HeadTransform.o: HeadTransform.cpp HeadTransform.h
	$(CXX) -c -fpic HeadTransform.cpp -O2 -fpic $(INCLUDES) -o HeadTransform.o

clean:
	rm -rf *.o _Code.so Code.py *.pyc __pycache__/