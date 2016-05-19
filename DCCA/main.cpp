#include <iostream>
#include <fstream>
#include <vector>

#include "DeepCCAModel.h"
#include "HyperParams.h"
#include "ProgramArgs.h"

using namespace std;

void ReadBin(const string & filename, AllocatingMatrix & mat, int numR, int maxCols = -1) {
	if (filename.size() == 0) return;

	ifstream inStream(filename, ios::in|ios::binary);
	if (!inStream.is_open()) {
		cout << "Couldn't open feature file " << filename.c_str() << endl;
		exit(1);
	}

	inStream.seekg(0, ios::end);
	int endPos = inStream.tellg();
	inStream.seekg(0, ios::beg);
	ASSERT(endPos / sizeof(double) % numR == 0);
	int numC = (int)(endPos / sizeof(double) / numR);

	if (maxCols != -1) numC = min(numC, maxCols);

	mat.Resize(numR, numC);
	inStream.read((char*)mat.Start(), numR * numC * sizeof(double));
	inStream.close();

	cout << "Read " << filename << " of size " << numR << "x" << numC << endl;
}

void LoadData(const ProgramArgs & args, vector<AllocatingMatrix> & trainData) {
	for (int v = 0; v < 2; ++v) {
		ReadBin(args.inData[v], trainData[v], args.iSize[v], args.trainSize);
	}

	double corr = CCA::TestCorr(trainData);
	if (!IsNaN(corr)) cout << "trainset linear corr: " << corr << endl << endl;
}

double Map(const DeepCCAModel & model, const vector<AllocatingMatrix> & data, const string outputData[]) {
	vector<AllocatingMatrix> mapped(2);

	for (int v = 0; v < 2; ++v) {
		if (data[v].Len() > 0) {
			model.MapUp(data[v], mapped[v], v);
			if (outputData[v].size() > 0) mapped[v].WriteToFile(outputData[v]);
		}
	}
	
	return CCA::TestCorr(mapped);
}

void Train(const ProgramArgs & args, DeepCCAModel & model, const vector<AllocatingMatrix> & trainData) {
	DCCAHyperParams hyperParams;
	Deserialize(hyperParams, args.inParams);

	// override params with command line if specified
	for (int v = 0; v < 2; ++v) {
		if (args.hSize[v] > 0) hyperParams.params[v].layerWidthH = args.hSize[v];
	}

	cout << endl << "Hyperparams: " << endl;
	hyperParams.Print();
	cout << endl;

	Random rand;

	TrainModifiers pretrainModifiers = TrainModifiers::LBFGSModifiers(1e-3, 15, false);
	TrainModifiers finetuneModifiers = TrainModifiers::LBFGSModifiers(1e-4, 15, false);

	double corr = model.Train(hyperParams, args.numLayers, args.inFeatSelect, args.outputSize, trainData, pretrainModifiers, finetuneModifiers, rand);

	cout << "Regularized train DCCA corr: " << corr << endl;
}

int main(int argc, char** argv) {
	ProgramArgs args(argc, argv);
	
	vector<AllocatingMatrix> trainData(2);

	DeepCCAModel model;
	if (args.inModel.size() > 0) {
		cout << "Reading model from " << args.inModel << endl;
		Deserialize(model, args.inModel);
		for (int v = 0; v < 2; ++v) args.iSize[v] = model.InputSize(v);
		LoadData(args, trainData);
	} else {
		LoadData(args, trainData);
		Train(args, model, trainData);

		if (args.outModel.size() > 0) {
			Serialize(model, args.outModel);
		}
	}

	double corr = Map(model, trainData, args.outData);
	if (!IsNaN(corr)) cout << "Total DCCA corr: " << corr << endl;

	return 0;
}
