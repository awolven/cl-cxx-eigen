#include <clcxx/clcxx.hpp>
#include <string>
#include "clcxx_eigen.hpp"

CLCXX_PACKAGE EIGEN(clcxx::Package& pack) {
  using EigenMat = EigenMatWrapper<Eigen::MatrixXd, double>;

  pack.defclass<EigenMat, true>("Matrix")
      .constructor<Eigen::Index, Eigen::Index>()
      .defmethod("m.resize",
          static_cast<void (EigenMat::*)(Eigen::Index, Eigen::Index)>(
              &EigenMat::resize))
      .defmethod("m.block", &EigenMat::getBlock)
      .defmethod("m.row", &EigenMat::getRow)
      .defmethod("m.col", &EigenMat::getCol)
      .defmethod("m.size", &EigenMat::size)
      .defmethod("m.rows", &EigenMat::rows)
      .defmethod("m.cols", &EigenMat::cols)
      .defmethod("m.get-at-index", &EigenMat::get)
      .defmethod(
          "m.set-at-index",
          static_cast<void (EigenMat::*)(Eigen::Index, Eigen::Index, double)>(
              &EigenMat::set))
      .defmethod(
          "m.set-matrix",
          static_cast<void (EigenMat::*)(const EigenMat&)>(&EigenMat::set))
      .defmethod("m.set-zero", &EigenMat::set0)
      .defmethod("m.set-ones", &EigenMat::set1)
      .defmethod("m.set-identity", &EigenMat::setId)
      .defmethod("%m.set-from-array", &EigenMat::setFromArray)
      .defmethod("m.trace", &EigenMat::trace)
      .defmethod("m.sum", &EigenMat::sum)
      .defmethod("m.prod", &EigenMat::prod)
      .defmethod("m.mean", &EigenMat::mean)
      .defmethod("m.norm", &EigenMat::norm)
      .defmethod("m.squared-norm", &EigenMat::squaredNorm)
      .defmethod("m.print", &EigenMat::print)
      .defmethod("m.scale", &EigenMat::scale)
      .defmethod("m.add-scalar",
          static_cast<void (EigenMat::*)(const double&)>(&EigenMat::add))
      .defmethod("m.add-mat", static_cast<void (EigenMat::*)(const EigenMat&)>(
          &EigenMat::add))
      .defmethod("m.multiply", &EigenMat::multiply)
      .defmethod("m.inverse", &EigenMat::inv)
      .defmethod("m.pseudoInverse", &EigenMat::pseudoInverse)
      .defmethod("m.full-inverse", &EigenMat::mInv)
      .defmethod("m.transpose", &EigenMat::trans)
      .defmethod("m.determinant", &EigenMat::det)
      .defmethod("m.full-determinant", &EigenMat::mDet)
      .defmethod("m.rank", &EigenMat::rankLU)
      .defmethod("m.full-q", &EigenMat::mQ)
      .defmethod("m.full-p", &EigenMat::mP)
      .defmethod("m.p", &EigenMat::squareMP)
      .defmethod("m.l", &EigenMat::squareML)
      .defmethod("m.u", &EigenMat::squareMU)
      .defmethod("m.lower-Cholesky", &EigenMat::mLCholesky)
      .defmethod("m.upper-Cholesky", &EigenMat::mUCholesky)
      .defmethod("m.eigen-values", &EigenMat::eigenVals)
      .defmethod("m.full-solve", &EigenMat::solveLU)
      .defmethod("m.solve", &EigenMat::solveSquareLU)
      .defmethod("m.solve-Cholesky", &EigenMat::solveCholesky)
      .defmethod("m.full-lower", &EigenMat::mU)
      .defmethod("m.full-upper", &EigenMat::mL)
      .defmethod("m.full-pivot-lu-solve", &EigenMat::fullPivLuSolve)
      .defmethod("m.infinity-lp-norm", &EigenMat::infinityLpNorm)
      .defmethod("m.one-lp-norm", &EigenMat::oneLpNorm)
      .defmethod("m.top-rows", &EigenMat::topRows)
      .defmethod("m.threshold-qr", &EigenMat::thresholdQR)
      .defmethod("m.set-threshold-qr", &EigenMat::setThresholdQR)
      .defmethod("m.compute-qr", &EigenMat::computeQR)
      .defmethod("m.qr-rows", &EigenMat::QR_rows)
      .defmethod("m.qr-cols", &EigenMat::QR_cols)
      .defmethod("m.qr-rank", &EigenMat::QR_rank)
      .defmethod("m.qr-top-rows-triangular-view-upper-transpose-solve-on-the-right", &EigenMat::QR_topRows_TriangularViewUpper_Transpose_SolveOnTheRight)
      .defmethod("m.qr-top-rows-triangular-view-upper", &EigenMat::QR_topRows_TriangularViewUpper)
      .defmethod("m.matrix-qr", &EigenMat::matrixQR)
      .defmethod("m.matrix-q", &EigenMat::matrixQ)
      .defmethod("m.cols-permutation-indices-row", &EigenMat::colsPermutationIndicesRow)
      .defmethod("m.cols-permutation", &EigenMat::colsPermutation)
      .defmethod("m.triangular-view-qr", &EigenMat::triangularViewQR)
      .defmethod("m.replicate", &EigenMat::replicate)
      .defmethod("m.scalemult", &EigenMat::scalemult)
      .defmethod("m.dot", &EigenMat::dot);

  pack.defun("m.m*", [](const EigenMat& x, const EigenMat& y) -> EigenMat {
    return static_cast<EigenMat>(x * y);
  });
  pack.defun("m.m+", [](const EigenMat& x, const EigenMat& y) -> EigenMat {
    return static_cast<EigenMat>(x + y);
  });
  pack.defun("m.mx", [](double x, const EigenMat& y) -> EigenMat {
      return static_cast<EigenMat>(x * y);
      });
  pack.defun("qp-eq", [](EigenMat& H, EigenMat& g, EigenMat& A, EigenMat& c, EigenMat& x, EigenMat& Y, EigenMat& Z) -> int {
      Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qrAT(A.transpose());
      Eigen::MatrixXd Q = qrAT.matrixQ();

      size_t params_num = qrAT.rows();
      size_t constr_num = qrAT.cols();
      size_t rank = qrAT.rank();

      if (rank != constr_num || constr_num > params_num)
          return -1;

      // A^T = Q*R*P^T = Q1*R1*P^T
      // Q = [Q1,Q2], R=[R1;0]
      // Y = Q1 * inv(R^T) * P^T
      // Z = Q2
      Y = qrAT.matrixQR().topRows(constr_num)
          .triangularView<Eigen::Upper>()
          .transpose()
          .solve<Eigen::OnTheRight>(Q.leftCols(rank))
          * qrAT.colsPermutation().transpose();
      if (params_num == rank)
          x = -Y * c;
      else {
          Z = Q.rightCols(params_num - rank);

          Eigen::MatrixXd ZTHZ = Z.transpose() * H * Z;
          Eigen::MatrixXd temp = (H * Y * c);
          Eigen::MatrixXd temp2 = (temp - g);
          Eigen::VectorXd rhs = Z.transpose() * temp2;

          Eigen::VectorXd y = ZTHZ.colPivHouseholderQr().solve(rhs);

          x = -Y * c + Z * y;
      }

      return 0;
      });

}
