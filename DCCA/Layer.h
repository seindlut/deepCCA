#pragma once

#include <assert.h>
#include <vector>

#include "Matrix.h"
#include "Random.h"

using namespace std;

class Layer {
public:
	enum ActivationType { TANH, CUBIC, LINEAR };

private:
	AllocatingMatrix _a;
	ActivationType _actType;
	
	void ComputeInputs(const Matrix & weights, const Vector & biases, const Matrix & values, bool trans) {
		int numIns = values.NumC();
		int size = trans ? weights.NumC() : weights.NumR();
		_a.Resize(size, numIns);

		_a.AddProd(weights, trans, values, false, 1.0, 0.0);
		for (int i = 0; i < numIns; ++i) _a.GetCol(i) += biases;
	}

	static double MySigmoid(double y) {
		bool negate = false;
		if (y < 0) { negate = true; y = -y; }
		double x = (y <= 20) ? y : pow(3 * y, 1.0/3);

		double newX;
		while (true) {
			double xSqr = x * x;
			newX = (0.66666666666666666 * xSqr * x + y) / (xSqr + 1.0);
			if (newX >= x) break;
			x = newX;
		}

		return negate ? -newX : newX;
	}

	void ComputeActivations(ActivationType actType) {
		_actType = actType;
		switch (actType) {
		case TANH: _a.Tanh(); break;
		case CUBIC:
#ifdef __LINUX
			{
				_a.Apply(MySigmoid);
			}
			break;
#else
			_a.ModCubeRootSigmoid8SSE(); break;
#endif

		case LINEAR: break;
		default: abort();
		}
	}

public:
	Layer() { }

	const Matrix & ActivateUp(const Matrix & weights, const Vector & biases, const Matrix & lowerValues, ActivationType actType) {
		ComputeInputs(weights, biases, lowerValues, true);
		ComputeActivations(actType);
		return _a;
	}
	
	const Matrix & ActivateUp(const vector<Matrix> & weights, const Vector & biases, const vector<Matrix> & lowerValues, ActivationType actType) {
		int numIns = lowerValues[0].NumC();
		int size = weights[0].NumC();
		_a.Resize(size, numIns);

		for (int i = 0; i < numIns; ++i) _a.GetCol(i).CopyFrom(biases);
		for (int l = 0; l < weights.size(); ++l) {
			_a.AddProd(weights[l], true, lowerValues[l], false);
		}

		ComputeActivations(actType);
		return _a;
	}
	
	double ActivateUpAndGetNegFreeEnergy(const Matrix & weights, const Vector & biases, const Matrix & lowerValues) {
		ComputeInputs(weights, biases, lowerValues, true);

		double energy = 0;
		auto func = [&](double aVal)->double {
			if (aVal < -14) {
				energy += aVal;
				return -1;
			} else if (aVal > 14) {
				energy += aVal;
				return 1;
			} else {
				double e = exp(aVal);
				double er = 1.0 / e;
				energy += log (e + er);
				return (e - er) / (e + er);
			}
		};
		_a.Apply(func);

		return energy;
	}
	
	double ActivateDownAndGetNegLL(const Matrix & weights, const Vector & biases, const Matrix & upperValues, const Matrix & lowerValues) {
		ComputeInputs(weights, biases, upperValues, false);

		double negll = 0;
		auto func = [&](double aVal, double xVal)->double {
			if (aVal < -14) {
				double xP = (1 + xVal) / 2;
				negll -= 2 * xP * aVal;
				return -1;
			} else if (aVal > 14) {
				double xP = (1 + xVal) / 2;
				negll += 2 * (1 - xP) * aVal;
				return 1;
			} else {
				double a = tanh(aVal);
				double p = (1 + a) / 2;
				double xP = (1 + xVal) / 2;
				negll -= xP * log(p) + (1 - xP) * log (1.0 - p);
				return a;
			}
		};
		_a.ApplyIntoRef(lowerValues, func);

		return negll;
	}
	
	const Matrix & ActivateDown(const Matrix & weights, const Vector & biases, const Matrix & upperValues, ActivationType actType) {
		ComputeInputs(weights, biases, upperValues, false);
		ComputeActivations(actType);
		return _a;
	}

	static void BackProp(const Matrix & weights, const Matrix & upperErrors, AllocatingMatrix & lowerInErrors) {
		lowerInErrors.Resize(weights.NumR(), upperErrors.NumC());
		lowerInErrors.AddProd(weights, false, upperErrors, false, 1.0, 0.0);
	}

	static void BackProp(const Matrix & weights, const Matrix & upperErrors, const Matrix & weights2, const Matrix & upperErrors2, AllocatingMatrix & lowerInErrors) {
		lowerInErrors.Resize(weights.NumR(), upperErrors.NumC());
		lowerInErrors.AddProd(weights, false, upperErrors, false, 1.0, 0.0);
		lowerInErrors.AddProd(weights2, false, upperErrors2, false);
	}

	static void ReverseBackProp(const Matrix & weights, const Matrix & lowerErrors, AllocatingMatrix & upperInErrors) {
		upperInErrors.Resize(weights.NumC(), lowerErrors.NumC());
		upperInErrors.AddProd(weights, true, lowerErrors, false, 1.0, 0.0);
	}

	const Matrix & ComputeErrors(const Matrix & inError) {
		switch (_actType) {
		case TANH:
			{
				auto func = [](double aVal, double eVal) { return (1.0 - aVal * aVal) * eVal; };
				_a.ApplyIntoRef(inError, func);
			}
			break;
		case CUBIC:
			{
				auto func = [](double aVal, double eVal) { return eVal / (1.0 + aVal * aVal); };
				_a.ApplyIntoRef(inError, func);
			}
			break;
		case LINEAR: _a.CopyFrom(inError); break;
		default: abort();
		}

		return _a;
	}

	MutableMatrix & Activations() { return _a; }

	int Size() const { return _a.NumR(); }

	int Count() const { return _a.NumC(); }
	
	void Sample(MutableMatrix & sample, Random & rand) const {
		auto func = [&](double aVal) { return (2 * rand.Uniform() - 1 < aVal ? 1.0 : -1.0); };
		sample.ApplyInto(_a, func);
	}

	void SampleGaussian(MutableMatrix & sample, double stdDev, Random & rand) const {
		auto func = [&](double aVal) { return aVal + stdDev * rand.Normal(); };
		sample.ApplyInto(_a, func);
	}
	
	void Clear() {
		_a.Resize(0, 0);
	}
};
