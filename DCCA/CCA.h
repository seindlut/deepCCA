#pragma once

#include "Matrix.h"
#include <assert.h>
#include <iostream>

#include "WhitenTransform.h"

class CCA {
	vector<AllocatingMatrix> _w;
	vector<AllocatingVector> _mu;

	static void GetMean(const Matrix & X, AllocatingVector & mu) {
		int m = X.NumC();
		AllocatingVector _ones(m, 1.0);
		mu.Resize(X.NumR());
		mu.MultInto(X, false, _ones, 1.0 / m, 0.0);
	}

	static void Translate(const Matrix & X, const Vector & mu, AllocatingMatrix & barX) {
		int m = X.NumC();
		AllocatingVector _ones(m, 1.0);
		barX.CopyFrom(X);
		barX.RankOneUpdate(mu, _ones, -1.0);
	}

public:
	CCA() : _w(2), _mu(2) { }

	class TraceNormObjective {
		vector<AllocatingMatrix> _barX;
		AllocatingMatrix _S11, _S12, _S22;
		AllocatingMatrix _nabla11, _nabla12, _nabla22;
		vector<AllocatingVector> _mu;
		AllocatingMatrix _U, _Vt;
		AllocatingVector _D;
		AllocatingVector _superB;
		vector<double> _lambda;

		// computes the Cholesky factor of [(HH' + rI)/(m-1)]^{1/2}
		// Sigma_ii^{1/2} in the paper
		void MakeSqrtCovCholesky(const Matrix & input, AllocatingMatrix & output, double lambda) {
			int n = input.NumR(), m = input.NumC();
			int k = min(n,m);

			_U.Resize(n, n);
			_U.Syrk(input, 1.0 / (m-1), 0.0);
			for (int i = 0; i < n; ++i) _U.At(i,i) += lambda / (m-1);

			_D.Resize(n);
			int info = LAPACKE_dsyevd(CblasColMajor, 'V', 'U', n, _U.Start(), n, _D.Start());

			if (info != 0) {
				cout << "dsyevd returned error code " << info << endl;
			}

			output.Resize(n,n);
			output.Clear();
			for (int i = 0; i < n; ++i) {
				output.SymmetricRankOneUpdate(_U.GetCol(i), sqrt(_D[i]));
			}

			info = LAPACKE_dpotrf(CblasColMajor, 'U', n, output.Start(), n);

			if (info != 0) {
				cout << "dpotrf returned error code " << info << endl;
			}
		}

	public:
		TraceNormObjective(const vector<double> & lambda) : _mu(2), _barX(2), _lambda(lambda) { }

		double EvalTrace(const Matrix & X1, const Matrix & X2, MutableMatrix & D1, MutableMatrix & D2) {
			const Matrix X[] = { X1, X2 };
			int n1 = X1.NumR(), n2 = X2.NumR();
			int k = min(n1, n2);
			int m = X1.NumC();
			assert (m == X2.NumC());

			bool shallowView1 = (D1.Len() == 0);
			bool shallowView2 = (D2.Len() == 0);
			
			for (int i = 0; i < 2; ++i) {
				GetMean(X[i], _mu[i]);
				Translate(X[i], _mu[i], _barX[i]);
			}

			_S12.Resize(n1, n2);
			_S12.AddProd(_barX[0], false, _barX[1], true, 1.0 / (m-1), 0.0);

			MakeSqrtCovCholesky(_barX[0], _S11, _lambda[0]);
			MakeSqrtCovCholesky(_barX[1], _S22, _lambda[1]);

			// set _S12 = S11^{-1/2} S12 S22^{-1/2}
			// first multiply by S22^{-1/2} on right
			LAPACKE_dpotrs(CblasRowMajor, 'L', n2, n1, _S22.Start(), n2, _S12.Start(), n1);
			// then multiply by S11^{-1/2} on left
			LAPACKE_dpotrs(CblasColMajor, 'U', n1, n2, _S11.Start(), n1, _S12.Start(), n1);

			_U.Resize(n1, k);
			_Vt.Resize(k, n2);

			// get SVD of S11^{-1/2} S12 S22^{-1/2}
			_D.Resize(k);
			_superB.Resize(k);
			int info = LAPACKE_dgesvd(CblasColMajor, 'S', 'S', n1, n2, _S12.Start(), n1, _D.Start(),
				_U.Start(), n1, _Vt.Start(), k, _superB.Start());

			if (info != 0) {
				cout << "dgesvd returned error code " << info << endl;
			}

			double val = _D.Sum();
			
			// put S11^{-1/2} U in _U
			LAPACKE_dpotrs(CblasColMajor, 'U', n1, k, _S11.Start(), n1, _U.Start(), n1);
			// put S11^{-1/2} V in _V
			LAPACKE_dpotrs(CblasRowMajor, 'L', n2, k, _S22.Start(), n2, _Vt.Start(), k);

			// form nabla12
			_nabla12.Resize(n1, n2);			
			_nabla12.AddProd(_U, false, _Vt, false, 1.0, 0.0);

			if (!shallowView1) {
				for (int i = 0; i < k; ++i) {
					_U.GetCol(i) *= sqrt(_D[i]);
				}
				_nabla11.Resize(n1, n1);
				cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, n1, k, -1.0/2, _U.Start(), n1, 0, _nabla11.Start(), n1);
				D1.AddProd(_nabla12, false, _barX[1], false, -1.0/2, 0.0);
				cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, n1, m, -1.0, _nabla11.Start(), n1, _barX[0].Start(), n1, 1.0, D1.Start(), n1);
			}

			if (!shallowView2) {
				for (int i = 0; i < k; ++i) {
					_Vt.GetRow(i) *= sqrt(_D[i]);
				}
				_nabla22.Resize(n2, n2);
				cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, n2, k, -1.0/2, _Vt.Start(), k, 0, _nabla22.Start(), n2);
				D2.AddProd(_nabla12, true, _barX[0], false, -1.0/2, 0.0);
				cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, n2, m, -1.0, _nabla22.Start(), n2, _barX[1].Start(), n2, 1.0, D2.Start(), n2);
			}

			return 0.5 * (m-1) * (k - val);
		}
	};
	
	template<class ArrayOfMatrices>
	static double TestCorr(const ArrayOfMatrices & X) {
		if (X[0].NumC() == 0 || X[1].NumC() == 0) return NaN;

		int m = X[0].NumC();
		assert (X[1].NumC() == m);

		int n[] = { X[0].NumR(), X[1].NumR() };

		vector<AllocatingMatrix> temp(2);
		for (int i = 0; i < 2; ++i) {
			// put centered in temp
			AllocatingVector mu(n[i]);
			GetMean(X[i], mu);
			Translate(X[i], mu, temp[i]);

			// compute SVD
			int k = min(n[0], m);
			AllocatingMatrix U(n[i], k), Vt(k, m);
			AllocatingVector singularValues(k), superb(k);

			int info = LAPACKE_dgesvd(CblasColMajor, 'S', 'S', n[i], m, temp[i].Start(), n[i], singularValues.Start(),
				U.Start(), n[i], Vt.Start(), k, superb.Start());

			if (info != 0) {
				cout << "dgesvd returned error code " << info << endl;
			}

			// put UV' in temp
			temp[i].AddProd(U, false, Vt, false, 1.0, 0.0);
		}

		AllocatingMatrix temp2(n[0], n[1]);
		temp2.AddProd(temp[0], false, temp[1], true, 1.0, 0.0);

		int k = min(n[0],n[1]);
		AllocatingVector singularValues(k);
		AllocatingVector superb(k);
		int info = LAPACKE_dgesvd(CblasColMajor, 'N', 'N', n[0], n[1], temp2.Start(), n[0], singularValues.Start(),
			0, n[0], 0, k, superb.Start());

		if (info != 0) {
			cout << "dgesvd returned error code " << info << endl;
		}

		return singularValues.Sum();
	}

	template<class ArrType1>
	void Map(ArrType1 & X) const {
		Map(X, X);
	}

	template<class ArrType1, class ArrType2>
	void Map(const ArrType1 & X, ArrType2 & mapped) const {
		AllocatingMatrix barX;
		for (int i = 0; i < 2; ++i) {
			Translate(X[i], _mu[i], barX);
			mapped[i].Resize(_w[i].NumR(), X[i].NumC());
			mapped[i].AddProd(_w[i], false, barX, false, 1.0, 0.0);
		}
	}

	void Map(const Matrix & X, int which, AllocatingMatrix & mapped) const {
		AllocatingMatrix barX;
		Translate(X, _mu[which], barX);

		mapped.Resize(_w[which].NumR(), X.NumC());
		mapped.AddProd(_w[which], false, barX, false, 1.0, 0.0);
	}

	void Serialize(ostream & outStream) const {
		for (int i = 0; i < 2; ++i) {
			_mu[i].Serialize(outStream);
			_w[i].Serialize(outStream);
		}
	}
	
	void Deserialize(istream & inStream) {
		for (int i = 0; i < 2; ++i) {
			_mu[i].Deserialize(inStream);
			_w[i].Deserialize(inStream);
		}
	}

	template <class ArrayOfMatrices>
	double InitWeights(const ArrayOfMatrices & X, double reg1 = 0, double reg2 = NaN) {
		if (IsNaN(reg2)) reg2 = reg1;

		int m = X[0].NumC();
		int n[] = { X[0].NumR(), X[1].NumR() };
		assert (X[1].NumC() == m);
		double reg[] = { reg1, reg2 };

		vector<AllocatingMatrix> U(2);
		vector<AllocatingMatrix> centered(2);

		for (int i = 0; i < 2; ++i) {
			GetMean(X[i], _mu[i]);
			Translate(X[i], _mu[i], centered[i]);

			U[i].Resize(n[i], n[i]);
			U[i].Syrk(centered[i], 1.0 / (m - 1), 0.0);
			double thisReg = max(reg[i] / m, 1e-8);
			for (int r = 0; r < n[i]; ++r) U[i].At(r,r) += thisReg;
			LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', n[i], U[i].Start(), n[i]);
		}

		AllocatingMatrix S12(n[0], n[1]);
		S12.AddProd(centered[0], false, centered[1], true, 1.0 / (m-1), 0.0);

		// set S12 = U[0]^{-1}' S12 U[1]^{-1}
		LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N', n[1], n[0], U[1].Start(), n[1], S12.Start(), n[0]);
		LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N', n[0], n[1], U[0].Start(), n[0], S12.Start(), n[0]);

		int k = min(n[0], n[1]);
		_w[0].Resize(n[0], k);
		_w[1].Resize(k, n[1]);
		AllocatingVector singularValues(k);
		AllocatingVector superb(k);
		int info = LAPACKE_dgesvd(CblasColMajor, 'S', 'S', n[0], n[1], S12.Start(), n[0], singularValues.Start(),
			_w[0].Start(), n[0], _w[1].Start(), k, superb.Start());

		if (info != 0) {
			cout << "dgesvd returned error code " << info << endl;
		}

		_w[0].TransposeInPlace();
		for (int i = 0; i < 2; ++i) LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', k, n[i], U[i].Start(), n[i], _w[i].Start(), k);

		return singularValues.Sum();
	}
};
