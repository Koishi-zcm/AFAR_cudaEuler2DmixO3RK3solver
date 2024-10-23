#ifndef d2GaussQuadrature_H
#define d2GaussQuadrature_H

#include "fvMesh.H"
#include "fvc.H"

namespace Foam
{

class d2GaussQuadrature
{
private:
	const fvMesh& mesh_;

	List<List<vector>> cellGaussPoints_;

	// cell gauss weights are used to evaluate cell average value
	// so they have been divided by the 2D face area of the cell
	List<List<scalar>> cellGaussWeights_;

	List<List<vector>> faceGaussPoints_;

	// face gauss weights are used to evaluate face flux
	// so here they include face area
	List<List<scalar>> faceGaussWeights_;

	List<List<List<vector>>> boundaryFaceGaussPointsList_;  // [patchi][facei][k]

	List<List<List<scalar>>> boundaryFaceGaussWeightsList_;  // [patchi][facei][k]

	const label cOrder_;

	const label fOrder_;

	void calcCellGaussQuadrature();

	void calcFaceGaussQuadrature();

	void sortRecPoints(List<vector>& points);

	void collectRecGaussQuadrature(
		const label cOrder,
		List<vector>& standardRecGaussPoints,
		List<scalar>& standardRecGaussWeights
	);

	void collectTriGaussQuadrature(
		const label cOrder,
		List<vector>& standardTriGaussPoints,
		List<scalar>& standardTriGaussWeights
	);

	void collectBarGaussQuadrature(
		const label fOrder,
		List<scalar>& standardBarGaussPoints,
		List<scalar>& standardBarGaussWeights
	);

public:
	d2GaussQuadrature(const fvMesh& mesh);

	//~d2GaussQuadrature();

	const List<List<vector>>& cellGaussPoints() const
	{
		return this->cellGaussPoints_;
	}

	const List<List<scalar>>& cellGaussWeights() const
	{
		return this->cellGaussWeights_;
	}

	const List<List<vector>>& faceGaussPoints() const
	{
		return this->faceGaussPoints_;
	}

	const List<List<scalar>>& faceGaussWeights() const
	{
		return this->faceGaussWeights_;
	}

	const List<List<List<vector>>>& boundaryFaceGaussPointsList() const
	{
		return this->boundaryFaceGaussPointsList_;
	}

	const List<List<List<scalar>>>& boundaryFaceGaussWeightsList() const
	{
		return this->boundaryFaceGaussWeightsList_;
	}
};

}  // end of namespace
#endif