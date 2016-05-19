#pragma once

#include <iostream>

#include "WhitenTransform.h"
#include "DBN.h"
#include "CCA.h"
#include "Matrix.h"
#include "HyperParams.h"

using namespace std;

class DeepCCAModel {
public:
	typedef DCCAHyperParams HyperParamType;

private:
	DCCAHyperParams _hyperParams;
	vector<WhitenTransform> _whiten;
	vector<DBN> _dbn;
	AllocatingVector _params;
	CCA _cca;

	MutableVector GetParams(int which) {
		assert (which * which == which);

		int numParams0 = _dbn[0].NumParams();
		return (which == 0) ? _params.SubVector(0, numParams0) : _params.SubVector(numParams0, -1);
	}
	
	class BackpropFunction : public DifferentiableFunction {
		CCA::TraceNormObjective _traceNormObjective;
		DBN & _dbn1, & _dbn2;
		const Matrix _input1, _input2;
		AllocatingMatrix _tempInput1, _tempInput2;
		int _numLayers1, _numLayers2;
		double _alpha;

		double PrivateEval(const Vector & params, MutableVector & gradient, const Matrix & inputMiniBatch1, const Matrix & inputMiniBatch2) {
			int numParams1 = _dbn1.NumParams(_numLayers1);
			_dbn1.SetReadParams(params.SubVector(0, numParams1), _numLayers1);
			_dbn2.SetReadParams(params.SubVector(numParams1, -1), _numLayers2);

			Matrix mappedInput1 = _dbn1.MapUp(inputMiniBatch1, _numLayers1), mappedInput2 = _dbn2.MapUp(inputMiniBatch2, _numLayers2);

			AllocatingMatrix & errors1 = _dbn1.GetTempMat();
			AllocatingMatrix & errors2 = _dbn2.GetTempMat();

			if (_numLayers1 > 0) errors1.Resize(mappedInput1.NumR(), mappedInput1.NumC());
			if (_numLayers2 > 0) errors2.Resize(mappedInput2.NumR(), mappedInput2.NumC());

			double val = _traceNormObjective.EvalTrace(mappedInput1, mappedInput2, errors1, errors2);
			if (!errors1.AllSafe() || !errors2.AllSafe()) {
				cout << "unsafe value in errors." << endl;
			}

			int m = inputMiniBatch1.NumC();
			val /= m;

			if (gradient.Len() > 0) {
				gradient.Clear();
				_dbn1.SetWriteParams(gradient.SubVector(0, numParams1), _numLayers1);
				_dbn2.SetWriteParams(gradient.SubVector(numParams1, -1), _numLayers2);
				_dbn1.BackProp(inputMiniBatch1, errors1, _numLayers1);
				_dbn2.BackProp(inputMiniBatch2, errors2, _numLayers2);				
				gradient /= m;
			} else {
				_dbn1.ClearWriteParams();
				_dbn2.ClearWriteParams();
			}

			// regularize
			double alphaEff = _alpha * m / NumIns();
			val += _dbn1.Regularize(alphaEff, _numLayers1);
			val += _dbn2.Regularize(alphaEff, _numLayers2);

			return val + 1;
		}
		
		MutableVector _emptyVector;

	public:
		BackpropFunction(const Matrix & input1, const Matrix & input2, DBN & dbn1, DBN & dbn2, const vector<double> & lambda, double alpha)
			:
		_traceNormObjective(lambda),
		_dbn1(dbn1), _dbn2(dbn2), _input1(input1), _input2(input2),
		_numLayers1(dbn1.NumLayers()), _numLayers2(dbn2.NumLayers()), _alpha(alpha)
		{ }

		void SetReadParams(const Vector & params) {
			int numParams1 = _dbn1.NumParams(_numLayers1);
			_dbn1.SetReadParams(params.SubVector(0, numParams1), _numLayers1);
			_dbn2.SetReadParams(params.SubVector(numParams1, -1), _numLayers2);
		}

		int NumParams() { return _dbn1.NumParams(_numLayers1) + _dbn2.NumParams(_numLayers2); }

		int NumIns() const { return _input1.NumC(); }
		
		double Eval(const Vector & params, MutableVector & gradient) {
			return PrivateEval(params, gradient, _input1, _input2);
		}
		
		double Eval(const Vector & params) {
			return PrivateEval(params, _emptyVector, _input1, _input2);
		}

		double Eval(const Vector & params, MutableVector & gradient, const vector<int> & indices) {
			int n = indices.size();

			_tempInput1.Resize(_input1.NumR(), n);
			_tempInput2.Resize(_input2.NumR(), n);

			for (int i = 0; i < n; ++i) {
				_tempInput1.GetCol(i).CopyFrom(_input1.GetCol(indices[i]));
				_tempInput2.GetCol(i).CopyFrom(_input2.GetCol(indices[i]));
			}
			
			return PrivateEval(params, gradient, _tempInput1, _tempInput2);
		}

		double Eval(const Vector & params, MutableVector & gradient, int start, int count) {
			int numIns = NumIns();
			if (count == -1) count = numIns;
			start %= numIns;

			Matrix input1, input2;
			int end = start + count;
			if (end <= NumIns()) {
				input1 = _input1.SubMatrix(start, end);
				input2 = _input2.SubMatrix(start, end);
			} else {
				_tempInput1.Resize(_input1.NumR(), count);
				_tempInput1.SubMatrix(0, (NumIns() - start)).CopyFrom(_input1.SubMatrix(start, -1));
				_tempInput1.SubMatrix(NumIns() - start, -1).CopyFrom(_input1.SubMatrix(0, end - NumIns()));
				input1 = _tempInput1;

				_tempInput2.Resize(_input2.NumR(), count);
				_tempInput2.SubMatrix(0, (NumIns() - start)).CopyFrom(_input2.SubMatrix(start, -1));
				_tempInput2.SubMatrix(NumIns() - start, -1).CopyFrom(_input2.SubMatrix(0, end - NumIns()));
				input2 = _tempInput2;
			}

			return PrivateEval(params, gradient, input1, input2);
		}
	};

public:
	DeepCCAModel() : _whiten(2), _dbn(2) { }
	DeepCCAModel(DCCAHyperParams hyperParams) : _whiten(2), _dbn(2), _hyperParams(hyperParams) { }
	
	void Deserialize(istream & inStream)
	{
		for (int i = 0; i < 2; ++i) {
			_whiten[i].Deserialize(inStream);
			_dbn[i].Deserialize(inStream);
		}

		_params.Deserialize(inStream);
		_dbn[0].SetReadParams(GetParams(0));
		_dbn[1].SetReadParams(GetParams(1));

		_cca.Deserialize(inStream);
	}

	void Serialize(ostream & outStream) const {
		for (int i = 0; i < 2; ++i) {
			_whiten[i].Serialize(outStream);
			_dbn[i].Serialize(outStream);
		}

		_params.Serialize(outStream);

		_cca.Serialize(outStream);
	}

	double Train(DCCAHyperParams hyperParams, const int numLayers[], const int inFeatSelect[], int outputSize, const vector<AllocatingMatrix> & trainData, TrainModifiers pretrainModifiers, TrainModifiers trainModifiers, Random & rand) {
		_hyperParams = hyperParams;

		vector<AllocatingMatrix> whitenedTrainData(2);
		for (int v = 0; v < 2; ++v) {
			_whiten[v].Init(trainData[v], inFeatSelect[v]);
			_whiten[v].Transform(trainData[v], whitenedTrainData[v]);

			int hSize = _hyperParams.params[v].GetLayerWidthH();
			_dbn[v].Initialize(numLayers[v], whitenedTrainData[v].NumR(), hSize, outputSize);
		}

		_params.Resize(_dbn[0].NumParams() + _dbn[1].NumParams());

		for (int v = 0; v < 2; ++v) {
			_dbn[v].Pretrain(whitenedTrainData[v], GetParams(v), rand, false, _hyperParams.params[v], pretrainModifiers);
		}

		vector<double> lambda(2);
		lambda[0] = _hyperParams.ccaReg1, lambda[1] = _hyperParams.ccaReg2;
		BackpropFunction backpropFunc(whitenedTrainData[0], whitenedTrainData[1], _dbn[0], _dbn[1], lambda, _hyperParams.backpropReg);

		LBFGS opt(false);
		opt.Minimize(backpropFunc, _params, _params, trainModifiers.LBFGS_tol, trainModifiers.LBFGS_M, trainModifiers.testGrad);

		backpropFunc.SetReadParams(_params);

		Matrix mappedData[2];

		mappedData[0] = _dbn[0].MapUp(whitenedTrainData[0]);
		mappedData[1] = _dbn[1].MapUp(whitenedTrainData[1]);

		return _cca.InitWeights(mappedData, _hyperParams.ccaReg1, _hyperParams.ccaReg2);
	}

	void MapUp(const Matrix & inData, AllocatingMatrix & outData, int which) const {
		Matrix mappedData;

		// using outData as a temp here
		_whiten[which].Transform(inData, outData);
		mappedData = _dbn[which].MapUp(outData);

		_cca.Map(mappedData, which, outData);
	}

	int InputSize(int view) const {
		return _whiten[view].InSize();
	}
};
