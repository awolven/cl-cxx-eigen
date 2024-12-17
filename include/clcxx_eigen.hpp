#pragma once

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <cstring>
#include <sstream>
#include <string>

template <typename EigenT, typename ElementT>
class EigenMatWrapper : public EigenT {
  using type = EigenMatWrapper<EigenT, ElementT>;
  using RowMatrix =
      Eigen::Matrix<ElementT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

 public:
  EigenMatWrapper() : EigenT() {}
  EigenMatWrapper(Eigen::Index rows, Eigen::Index cols) {
    *this = this->Zero(rows, cols);
  }
  // This constructor allows you to construct MyVectorType from Eigen
  // expressions
  explicit EigenMatWrapper(const EigenT& obj) : EigenT(obj) {}

  // This method allows you to assign Eigen expressions to MyVectorType
  type& operator=(const EigenT& other) {
    this->EigenT::operator=(other);
    return *this;
  }

  void set(Eigen::Index i, Eigen::Index j, ElementT value) {
    if ((this->rows() > i) && (this->cols() > j) && (i > -1) && (j > -1)) {
      this->operator()(i, j) = value;
    } else {
      throw std::runtime_error("Wrong index");
    }
  }
  void set(const type& obj) { this->EigenT::operator=(obj); }
  void set0() { this->setZero(); }
  void setId() { this->setIdentity(); }
  void set1() { this->setOnes(); }
  void getCol(Eigen::Index i) { this->col(i); }
  void getRow(Eigen::Index i) { this->row(i); }
  type getBlock(Eigen::Index i, Eigen::Index j, Eigen::Index k,
                Eigen::Index l) {
    return static_cast<type>(this->block(i, j, k, l));
  }
  ElementT get(Eigen::Index i, Eigen::Index j) {
    if ((this->rows() > i) && (this->cols() > j) && (i > -1) && (j > -1)) {
      return this->operator()(i, j);
    } else {
      throw std::runtime_error("Wrong index");
    }
  }
  std::string print() {
    std::stringstream s;
    s << *this;
    return s.str();
  }
  void scale(const ElementT& value) { *this *= value; }
  void add(const ElementT& value) {
    this->unaryExpr([value](double x) { return x + value; });
  }
  void add(const type& obj) { *this += (type)obj; }
  void multiply(const type& obj) { *this *= obj; }
  type inv() { return static_cast<type>(this->inverse()); }
  type trans() { return static_cast<type>(this->transpose()); }
  ElementT det() { return this->determinant(); }
  ElementT dot(const type& m) { return Eigen::VectorXd(*this).dot(Eigen::VectorXd(m)); }
  ElementT thresholdQR() {
      return static_cast<Eigen::FullPivHouseholderQR<EigenT>>(*this).threshold();
  }
  void setThresholdQR(ElementT threshold) {
      static_cast<Eigen::FullPivHouseholderQR<EigenT>>(*this).setThreshold(threshold);
  }
  type computeQR() {
      Eigen::FullPivHouseholderQR<EigenT> qr;
      qr.compute(*this);
      return static_cast<type>(qr.matrixQR());
  }
  type matrixQR() {
      return static_cast<type>(static_cast<Eigen::FullPivHouseholderQR<EigenT>>(*this).matrixQR());
  }
  Eigen::Index QR_rows() {
      return static_cast<Eigen::FullPivHouseholderQR<EigenT>>(*this).rows();
  }
  Eigen::Index QR_cols() {
      return static_cast<Eigen::FullPivHouseholderQR<EigenT>>(*this).cols();
  }
  Eigen::Index QR_rank() {
      return static_cast<Eigen::FullPivHouseholderQR<EigenT>>(*this).rank();
  }
  type QR_topRows_TriangularViewUpper_Transpose_SolveOnTheRight (Eigen::Index rows, Eigen::Index cols, const type& m) {
      return type(static_cast<EigenT>(static_cast<EigenT>(static_cast<Eigen::FullPivHouseholderQR<EigenT>>(*this).matrixQR()).topRows(rows)).triangularView<Eigen::Upper>().transpose().solve<Eigen::OnTheRight>((EigenT)m.leftCols(cols)));
  }
  type QR_topRows_TriangularViewUpper(Eigen::Index rows) {
      return type(static_cast<EigenT>(*this).topRows(rows).triangularView<Eigen::Upper>());
  }
  type matrixQ() {
      return static_cast<type>(static_cast<Eigen::FullPivHouseholderQR<EigenT>>(*this).matrixQ());
  }
  type triangularViewQR() {
      return static_cast<type>((*this).template triangularView<Eigen::Upper>());
  }
  type colsPermutation() {
      return static_cast<type>(static_cast<Eigen::FullPivHouseholderQR<EigenT>>(*this).colsPermutation());
  }
  Eigen::Index colsPermutationIndicesRow(int row) {
      return static_cast<Eigen::FullPivHouseholderQR<EigenT>>(*this).colsPermutation().indices()[row];
  }



  type topRows(int count) { return static_cast<type>((static_cast<EigenT>(*this)).topRows(count)); }

  Eigen::Index rankLU() {
    Eigen::FullPivLU<EigenT> lu(*this);
    return lu.rank();
  }
  type mL() {
    Eigen::FullPivLU<EigenT> lu(*this);
    type l = static_cast<type>(this->Identity(this->rows(), this->cols()));
    l.block(0, 0, this->rows(), this->cols())
        .template triangularView<Eigen::StrictlyLower>() = lu.matrixLU();
    return l;
  }
  type mU() {
    Eigen::FullPivLU<EigenT> lu(*this);
    return static_cast<type>(
        lu.matrixLU().template triangularView<Eigen::Upper>());
  }
  type mQ() {
    Eigen::FullPivLU<EigenT> lu(*this);
    return static_cast<type>(lu.permutationQ());
  }
  type mInv() {
    Eigen::FullPivLU<EigenT> lu(*this);
    return static_cast<type>(lu.inverse());
  }
  ElementT mDet() {
    Eigen::FullPivLU<EigenT> lu(*this);
    return lu.determinant();
  }
  type mP() {
    Eigen::FullPivLU<EigenT> lu(*this);
    return static_cast<type>(lu.permutationP());
  }
  type squareML() {
    type l = static_cast<type>(this->Identity(this->rows(), this->cols()));
    l.block(0, 0, this->rows(), this->cols())
        .template triangularView<Eigen::StrictlyLower>() =
        this->lu().matrixLU();
    return l;
  }
  type squareMU() {
    return static_cast<type>(
        this->lu().matrixLU().template triangularView<Eigen::Upper>());
  }
  type squareMP() { return static_cast<type>(this->lu().permutationP()); }
  type mLCholesky() { return static_cast<type>(this->llt().matrixL()); }
  type mUCholesky() { return static_cast<type>(this->llt().matrixU()); }
  std::string eigenVals() {
    Eigen::VectorXcd eivals = this->eigenvalues();
    std::stringstream s;
    s << eivals;
    return s.str();
  }
  // Ax = b ,returns x
  type solveCholesky(const type& b) {
    return static_cast<type>(this->llt().solve(b));
  }
  type solveLU(const type& b) {
    Eigen::FullPivLU<EigenT> lu(*this);
    return static_cast<type>(lu.solve(b));
  }
  type solveSquareLU(const type& b) {
    return static_cast<type>(this->lu().solve(b));
  }
  void setFromArray(ElementT arr[], Eigen::Index i, Eigen::Index j) {
    *this = Eigen::Map<RowMatrix>(arr, i, j);
  }
  type fullPivLuSolve(type b) {
      return static_cast<type>(static_cast<EigenT>(*this).fullPivLu().solve(static_cast<EigenT>(b)));
  }
  double infinityLpNorm() {
      //EigenT matrix(*this);
      return (double)((EigenT)(*this)).lpNorm<Eigen::Infinity>();
  }
  double oneLpNorm() {
      //EigenT matrix(*this);
      return (double)((EigenT)(*this)).lpNorm<1>();
  }
  type replicate() {
      return static_cast<type>(static_cast<EigenT>(*this).replicate(1,1));
  }
  type scalemult(ElementT& s) {
      return type(s * *this);
  }
  type pseudoInverse() {
      return static_cast<type>(static_cast<EigenT>(*this).completeOrthogonalDecomposition().pseudoInverse());
  }
};
