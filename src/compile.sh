g++ -std=c++0x -Wall -lMinuit main.C Sigmoid.cc SigmoidGradient.cc ProduceInputMatrix.cc Prediction.cc CostAndGrad.cc -o main `root-config  --cflags --evelibs` -larmadillo 
g++ -std=c++0x -Wall -lMinuit Preprocess.C -o preprocess `root-config  --cflags --evelibs`
