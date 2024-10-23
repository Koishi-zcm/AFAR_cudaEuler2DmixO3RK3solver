#include "d2GaussQuadrature.H"

Foam::d2GaussQuadrature::d2GaussQuadrature(const fvMesh& mesh)
:
	mesh_(mesh),
	cellGaussPoints_(mesh.nCells()),
	cellGaussWeights_(mesh.nCells()),
	faceGaussPoints_(mesh.nInternalFaces()),
	faceGaussWeights_(mesh.nInternalFaces()),
	boundaryFaceGaussPointsList_(mesh.C().boundaryField().size()),
	boundaryFaceGaussWeightsList_(mesh.C().boundaryField().size()),
	cOrder_(mesh.schemes().subDict("gaussQuadrature").lookupOrDefault("cOrder", 5)),
	fOrder_(mesh.schemes().subDict("gaussQuadrature").lookupOrDefault("fOrder", 4))
{
    // determine global gauss points and gauss weights for every cell
    calcCellGaussQuadrature();

    // evaluate global gauss points and weights for every face
    calcFaceGaussQuadrature();
}


void Foam::d2GaussQuadrature::calcCellGaussQuadrature()
{
    const pointField& pts = this->mesh_.points();
    const faceList& fcs = this->mesh_.faces();
    const scalarField& cellVolume = this->mesh_.V();

    forAll(this->mesh_.cells(), celli)
    {
        const cell& cc = this->mesh_.cells()[celli];
        const labelList pLabels = cc.labels(fcs);
        List<vector> points;

        // 2 dimensional situtation with x,y valid direction
        // find the points that construct the face
        scalar cellHeight = 0.0;
        scalar z = 0.0;
        forAll(pLabels, i)
        {
            const label pointi = pLabels[i];
            const vector pointPosition = pts[pointi];

            // use the position of the first point as reference layer
            if (i == 0)
            {
                z = pointPosition.z();
                points.append(pts[pointi]);
                continue;
            }

            // check if the points is on the reference layer
            if (mag(z - pointPosition.z()) < SMALL)
            {
                points.append(pts[pointi]);
            }
            else  // different layer
            {
            	cellHeight = mag(z - pointPosition.z());
            }
        }

        const scalar faceArea = cellVolume[celli]/cellHeight;

        if (points.size() == 3)
        {
			List<vector> standardTriGaussPoints;
			List<scalar> standardTriGaussWeights;
			collectTriGaussQuadrature(this->cOrder_, standardTriGaussPoints, standardTriGaussWeights);

        	List<vector>& cellGaussPoints = this->cellGaussPoints_[celli];
        	List<scalar>& cellGaussWeights = this->cellGaussWeights_[celli];
        	cellGaussPoints.setSize(standardTriGaussPoints.size());
        	cellGaussWeights.setSize(standardTriGaussWeights.size());

        	const scalar& x1 = points[0].x();
        	const scalar& x2 = points[1].x();
        	const scalar& x3 = points[2].x();
        	const scalar& y1 = points[0].y();
        	const scalar& y2 = points[1].y();
        	const scalar& y3 = points[2].y();

        	forAll(standardTriGaussPoints, k)
        	{
        		const scalar& xi = standardTriGaussPoints[k].x();
        		const scalar& eta = standardTriGaussPoints[k].y();

        		vector finalPoint = vector(0.0, 0.0, 0.0);
        		finalPoint.x() = x1 + (x2 - x1)*xi + (x3 - x1)*eta;
        		finalPoint.y() = y1 + (y2 - y1)*xi + (y3 - y1)*eta;
        		cellGaussPoints[k] = finalPoint;

        		const scalar jacobi = mag((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1));
        		// face area of the cell has been divided here
        		cellGaussWeights[k] = standardTriGaussWeights[k]*jacobi/faceArea;
        	}
        }
        else if (points.size() == 4)
        {
			List<vector> standardRecGaussPoints;
			List<scalar> standardRecGaussWeights;
			collectRecGaussQuadrature(this->cOrder_, standardRecGaussPoints, standardRecGaussWeights);

        	List<vector>& cellGaussPoints = this->cellGaussPoints_[celli];
        	List<scalar>& cellGaussWeights = this->cellGaussWeights_[celli];
        	cellGaussPoints.setSize(standardRecGaussPoints.size());
        	cellGaussWeights.setSize(standardRecGaussWeights.size());

        	sortRecPoints(points);

        	const scalar x1 = points[0].x();
        	const scalar x2 = points[1].x();
        	const scalar x3 = points[2].x();
        	const scalar x4 = points[3].x();
        	const scalar y1 = points[0].y();
        	const scalar y2 = points[1].y();
        	const scalar y3 = points[2].y();
        	const scalar y4 = points[3].y();

        	forAll(standardRecGaussPoints, k)
        	{
        		const scalar& xi = standardRecGaussPoints[k].x();
        		const scalar& eta = standardRecGaussPoints[k].y();

        		vector finalPoint = vector(0.0, 0.0, 0.0);
        		const scalar basis1 = 0.25*(1.0 - xi)*(1.0 - eta);
        		const scalar basis2 = 0.25*(1.0 - xi)*(1.0 + eta);
        		const scalar basis3 = 0.25*(1.0 + xi)*(1.0 + eta);
        		const scalar basis4 = 0.25*(1.0 + xi)*(1.0 - eta);
        		finalPoint.x() = x1*basis1 + x2*basis2 + x3*basis3 + x4*basis4;
        		finalPoint.y() = y1*basis1 + y2*basis2 + y3*basis3 + y4*basis4;
        		cellGaussPoints[k] = finalPoint;

        		const scalar dxdxi = -0.25*x1*(1.0 - eta) - 0.25*x2*(1.0 + eta) + 0.25*x3*(1.0 + eta) + 0.25*x4*(1.0 - eta);
        		const scalar dxdeta = -0.25*x1*(1.0 - xi) + 0.25*x2*(1.0 - xi) + 0.25*x3*(1.0 + xi) - 0.25*x4*(1.0 + xi);
        		const scalar dydxi = -0.25*y1*(1.0 - eta) - 0.25*y2*(1.0 + eta) + 0.25*y3*(1.0 + eta) + 0.25*y4*(1.0 - eta);
        		const scalar dydeta = -0.25*y1*(1.0 - xi) + 0.25*y2*(1.0 - xi) + 0.25*y3*(1.0 + xi) - 0.25*y4*(1.0 + xi);
        		const scalar jacobi = mag(dxdxi*dydeta - dxdeta*dydxi);
        		// face area of the cell has been divided here
        		cellGaussWeights[k] = standardRecGaussWeights[k]*jacobi/faceArea;
        	}
        }
        else
        {
        	FatalErrorIn("Foam::d2GaussQuadrature::calcCellGaussQuadrature()")
        		<< "problem cell appear with 2D face points number = " << points.size() << nl
        		<< exit(FatalError);
        }
    }
}


void Foam::d2GaussQuadrature::calcFaceGaussQuadrature()
{
	List<scalar> standardBarGaussPoints;
	List<scalar> standardBarGaussWeights;
	collectBarGaussQuadrature(this->fOrder_, standardBarGaussPoints, standardBarGaussWeights);

	const pointField& pts = this->mesh_.points();
	const faceList& fcs = this->mesh_.faces();
	const unallocLabelList& owner = this->mesh_.owner();

	forAll(owner, faceI)
	{
		const face& cf = fcs[faceI];
		const pointField& facePoints = cf.points(pts);
		List<vector> points;

		scalar z = 0.0;  // reference z position
		scalar faceHeight = 0.0;
		forAll(facePoints, i)
		{
			const vector& position = facePoints[i];

			// use the position of the first point as reference layer
			if (i == 0)
			{
				z = position.z();
				points.append(position);
				continue;
			}

			// check if the point is on the reference layer
			if (mag(z - position.z()) < SMALL)
			{
				points.append(position);
			}
			else
			{
				// face height for gauss weight correction later
				faceHeight = mag(z - position.z());
			}
		}

		const scalar jacobi = Foam::sqrt(
			sqr(0.5*(points[0].x() - points[1].x()))
			+ sqr(0.5*(points[0].y() - points[1].y()))
			);

		List<vector>& faceGaussPoints = this->faceGaussPoints_[faceI];
		List<scalar>& faceGaussWeights = this->faceGaussWeights_[faceI];
		faceGaussPoints.setSize(standardBarGaussPoints.size());
		faceGaussWeights.setSize(standardBarGaussWeights.size());

		forAll(standardBarGaussPoints, k)
		{
			const scalar& xi = standardBarGaussPoints[k];

			faceGaussPoints[k].x() = 0.5*points[0].x()*(1 - xi) + 0.5*points[1].x()*(1 + xi);
			faceGaussPoints[k].y() = 0.5*points[0].y()*(1 - xi) + 0.5*points[1].y()*(1 + xi);
			// face height is included here to match the genernal face flux calculation in OpenFOAM
			faceGaussWeights[k] = standardBarGaussWeights[k]*jacobi*faceHeight;
		}
	}


	forAll(this->mesh_.C().boundaryField(), patchi)
	{
		const fvPatchVectorField& pCellCentre = this->mesh_.C().boundaryField()[patchi];
		const fvPatch& curPatch = pCellCentre.patch();
		List<List<vector>>& patchFaceGaussPoints = this->boundaryFaceGaussPointsList_[patchi];
		List<List<scalar>>& patchFaceGaussWeights = this->boundaryFaceGaussWeightsList_[patchi];
		patchFaceGaussPoints.setSize(pCellCentre.size());
		patchFaceGaussWeights.setSize(pCellCentre.size());

		forAll(curPatch, facei)
		{
			const label faceI = curPatch.start() + facei;
			const face& cf = fcs[faceI];
			const pointField& facePoints = cf.points(pts);
			List<vector> points;

			scalar z = 0.0;  // reference z position
			scalar faceHeight = 0.0;
			forAll(facePoints, i)
			{
				const vector& position = facePoints[i];

				// use the position of the first point as reference layer
				if (i == 0)
				{
					z = position.z();
					points.append(position);
					continue;
				}

				// check if the point is on the reference layer
				if (mag(z - position.z()) < SMALL)
				{
					points.append(position);
				}
				else
				{
					// face height for gauss weight correction later
					faceHeight = mag(z - position.z());
				}
			}

			const scalar jacobi = Foam::sqrt(
				sqr(0.5*(points[0].x() - points[1].x()))
				+ sqr(0.5*(points[0].y() - points[1].y()))
				);

			List<vector>& faceGaussPoints = patchFaceGaussPoints[facei];
			List<scalar>& faceGaussWeights = patchFaceGaussWeights[facei];
			faceGaussPoints.setSize(standardBarGaussPoints.size());
			faceGaussWeights.setSize(standardBarGaussWeights.size());

			forAll(standardBarGaussPoints, k)
			{
				const scalar& xi = standardBarGaussPoints[k];

				faceGaussPoints[k].x() = 0.5*points[0].x()*(1 - xi) + 0.5*points[1].x()*(1 + xi);
				faceGaussPoints[k].y() = 0.5*points[0].y()*(1 - xi) + 0.5*points[1].y()*(1 + xi);
				// face height is included here to match the genernal face flux calculation in OpenFOAM
				faceGaussWeights[k] = standardBarGaussWeights[k]*jacobi*faceHeight;
			}
		}
	}
}


void Foam::d2GaussQuadrature::sortRecPoints(List<vector>& points)
{
	// use the geometry center point as the reference point for polar angle evaluation
	vector refPoint = vector(0.0, 0.0, 0.0);
	forAll(points, i)
	{
		refPoint += points[i];
	}
	refPoint /= scalar(points.size());

	// evaluate polar angle value for every point by reference point
	List<scalar> polar(points.size());
	forAll(polar, i)
	{
		const scalar x = points[i].x() - refPoint.x();
		const scalar y = points[i].y() - refPoint.y();
		const scalar r = Foam::sqrt(sqr(x) + sqr(y));
        const scalar angle = y >= 0 ? Foam::acos(x/r) : - Foam::acos(x/r);
        polar[i] = angle;
	}

	// sort points clockwise by polar angle value, the first item has largest angle
    labelList idx(polar.size());
    forAll(idx, i) idx[i] = i;  // initialize
    // bubble sort
    for (label i = 0; i < idx.size()-1; ++i)
    {
        for (label j = 0; j < idx.size()-i-1; ++j)
        {
            if (polar[j] < polar[j+1])
            {
                const scalar tmpPolar = polar[j];
                polar[j] = polar[j+1];
                polar[j+1] = tmpPolar;

                const label tmpIdx = idx[j];
                idx[j] = idx[j+1];
                idx[j+1] = tmpIdx;
            }
        }
    }

    List<vector> tmpPoints(4);
    tmpPoints[0] = points[idx[0]];
    tmpPoints[1] = points[idx[1]];
    tmpPoints[2] = points[idx[2]];
    tmpPoints[3] = points[idx[3]];
    points = tmpPoints;
}

void Foam::d2GaussQuadrature::collectRecGaussQuadrature(
	const label cOrder,
	List<vector>& standardRecGaussPoints,
	List<scalar>& standardRecGaussWeights
)
{
	// gauss points for standard square
	// four points are A(1,1) B(-1,1) C(-1,-1) D(1,-1) and the area is 4

	if (cOrder == 1)
	{
		standardRecGaussPoints.setSize(1);
		standardRecGaussWeights.setSize(1);
		List<vector> ps(1);
		List<scalar> ws(1);
		ps[0] = vector(0.0, 0.0, 0.0);  ws[0] = 4.0;
		standardRecGaussPoints = ps;
		standardRecGaussWeights = ws;
	}
	else if (cOrder > 1 && cOrder <= 3)
	{
		standardRecGaussPoints.setSize(4);
		standardRecGaussWeights.setSize(4);
		List<vector> ps(4);
		List<scalar> ws(4);
		ps[0] = vector(-1.0/Foam::sqrt(3.0), -1.0/Foam::sqrt(3.0), 0.0);  ws[0] = 1.0;
		ps[1] = vector(-1.0/Foam::sqrt(3.0), 1.0/Foam::sqrt(3.0), 0.0);  ws[0] = 1.0;
		ps[2] = vector(1.0/Foam::sqrt(3.0), -1.0/Foam::sqrt(3.0), 0.0);  ws[0] = 1.0;
		ps[3] = vector(1.0/Foam::sqrt(3.0), 1.0/Foam::sqrt(3.0), 0.0);  ws[0] = 1.0;
		standardRecGaussPoints = ps;
		standardRecGaussWeights = ws;
	}
	else if (cOrder > 3 && cOrder <= 5)
	{
		standardRecGaussPoints.setSize(9);
		standardRecGaussWeights.setSize(9);
		List<vector> ps(9);
		List<scalar> ws(9);
		ps[0] = vector(-Foam::sqrt(0.6), -Foam::sqrt(0.6), 0.0);  ws[0] = (5.0/9.0)*(5.0/9.0);
		ps[1] = vector(-Foam::sqrt(0.6), 0.0, 0.0);               ws[1] = (5.0/9.0)*(8.0/9.0);
		ps[2] = vector(-Foam::sqrt(0.6), Foam::sqrt(0.6), 0.0);   ws[2] = (5.0/9.0)*(5.0/9.0);
		ps[3] = vector(0.0, -Foam::sqrt(0.6), 0.0);               ws[3] = (8.0/9.0)*(5.0/9.0);
		ps[4] = vector(0.0, 0.0, 0.0);                            ws[4] = (8.0/9.0)*(8.0/9.0);
		ps[5] = vector(0.0, Foam::sqrt(0.6), 0.0);                ws[5] = (8.0/9.0)*(5.0/9.0);
		ps[6] = vector(Foam::sqrt(0.6), -Foam::sqrt(0.6), 0.0);   ws[6] = (5.0/9.0)*(5.0/9.0);
		ps[7] = vector(Foam::sqrt(0.6), 0.0, 0.0);                ws[7] = (5.0/9.0)*(8.0/9.0);
		ps[8] = vector(Foam::sqrt(0.6), Foam::sqrt(0.6), 0.0);    ws[8] = (5.0/9.0)*(5.0/9.0);
		standardRecGaussPoints = ps;
		standardRecGaussWeights = ws;
	}
	else if (cOrder > 5 && cOrder <= 7)
	{
		standardRecGaussPoints.setSize(16);
		standardRecGaussWeights.setSize(16);
		List<vector> ps(16);
		List<scalar> ws(16);
		ps[0] = vector(0.861136311594053, 0.861136311594053, 0.0);  ws[0] = 0.347854845137454*0.347854845137454;
		ps[1] = vector(0.339981043584856, 0.861136311594053, 0.0);  ws[1] = 0.652145154862546*0.347854845137454;
		ps[2] = vector(-0.339981043584856, 0.861136311594053, 0.0);  ws[2] = 0.652145154862546*0.347854845137454;
		ps[3] = vector(-0.861136311594053, 0.861136311594053, 0.0);  ws[3] = 0.347854845137454*0.347854845137454;

		ps[4] = vector(0.861136311594053, 0.339981043584856, 0.0);  ws[4] = 0.347854845137454*0.652145154862546;
		ps[5] = vector(0.339981043584856, 0.339981043584856, 0.0);  ws[5] = 0.652145154862546*0.652145154862546;
		ps[6] = vector(-0.339981043584856, 0.339981043584856, 0.0);  ws[6] = 0.652145154862546*0.652145154862546;
		ps[7] = vector(-0.861136311594053, 0.339981043584856, 0.0);  ws[7] = 0.347854845137454*0.652145154862546;

		ps[8] = vector(0.861136311594053, -0.339981043584856, 0.0);  ws[8] = 0.347854845137454*0.652145154862546;
		ps[9] = vector(0.339981043584856, -0.339981043584856, 0.0);  ws[9] = 0.652145154862546*0.652145154862546;
		ps[10] = vector(-0.339981043584856, -0.339981043584856, 0.0);  ws[10] = 0.652145154862546*0.652145154862546;
		ps[11] = vector(-0.861136311594053, -0.339981043584856, 0.0);  ws[11] = 0.347854845137454*0.652145154862546;

		ps[12] = vector(0.861136311594053, -0.861136311594053, 0.0);  ws[12] = 0.347854845137454*0.347854845137454;
		ps[13] = vector(0.339981043584856, -0.861136311594053, 0.0);  ws[13] = 0.652145154862546*0.347854845137454;
		ps[14] = vector(-0.339981043584856, -0.861136311594053, 0.0);  ws[14] = 0.652145154862546*0.347854845137454;
		ps[15] = vector(-0.861136311594053, -0.861136311594053, 0.0);  ws[15] = 0.347854845137454*0.347854845137454;
		standardRecGaussPoints = ps;
		standardRecGaussWeights = ws;
	}
	else
	{
		FatalErrorIn("Foam::d2GaussQuadrature::collectRecGaussQuadrature()")
			<< "No stored information for cOrder = " << cOrder << nl
			<< exit(FatalError);
	}
}



void Foam::d2GaussQuadrature::collectTriGaussQuadrature(
	const label cOrder,
	List<vector>& standardTriGaussPoints,
	List<scalar>& standardTriGaussWeights
)
{
	// gauss points for standard isosceles right triangle
	// three points are A(0,0) B(0,1) C(1,0) and the area is 0.5

	if (cOrder == 1)  // accuracy grade = 1
	{
		standardTriGaussPoints.setSize(1);
		standardTriGaussWeights.setSize(1);
		List<vector> ps(1);
		List<scalar> ws(1);
		ps[0] = vector(1.0/3.0, 1.0/3.0, 0.0);  ws[0] = 0.5;
		standardTriGaussPoints = ps;
		standardTriGaussWeights = ws;
	}
	else if (cOrder > 1 && cOrder <= 3)  // accuracy grade = 3
	{
		standardTriGaussPoints.setSize(3);
		standardTriGaussWeights.setSize(3);
		List<vector> ps(3);
		List<scalar> ws(3);
		ps[0] = vector(0.5, 0.5, 0.0);  ws[0] = 1.0/6.0;
		ps[1] = vector(0.5, 0.0, 0.0);  ws[1] = 1.0/6.0;
		ps[2] = vector(0.0, 0.5, 0.0);  ws[2] = 1.0/6.0;
		standardTriGaussPoints = ps;
		standardTriGaussWeights = ws;
	}
	else if (cOrder > 3 && cOrder <= 5)  // accuracy grade = 5
	{
		standardTriGaussPoints.setSize(4);
		standardTriGaussWeights.setSize(4);
		List<vector> ps(4);
		List<scalar> ws(4);
		ps[0] = vector(1.0/3.0, 1.0/3.0, 0.0);  ws[0] = 21.0/96.0;
		ps[1] = vector(3.0/5.0, 1.0/5.0, 0.0);  ws[1] = 25.0/96.0;
		ps[2] = vector(1.0/5.0, 1.0/5.0, 0.0);  ws[2] = 25.0/96.0;
		ps[3] = vector(1.0/5.0, 3.0/5.0, 0.0);  ws[3] = 25.0/96.0;
		standardTriGaussPoints = ps;
		standardTriGaussWeights = ws;
	}
	else if (cOrder > 5 && cOrder <= 7)  // accuracy grade = 7
	{
		standardTriGaussPoints.setSize(6);
		standardTriGaussWeights.setSize(6);
		List<vector> ps(6);
		List<scalar> ws(6);
		ps[0] = vector(0.816847572980459, 0.091576213509771, 0.0);  ws[0] = 0.109951743655322*0.5;
		ps[1] = vector(0.091576213509771, 0.816847572980459, 0.0);  ws[1] = 0.109951743655322*0.5;
		ps[2] = vector(0.091576213509771, 0.091576213509771, 0.0);  ws[2] = 0.109951743655322*0.5;
		ps[3] = vector(0.108103018168070, 0.445948490915965, 0.0);  ws[3] = 0.223381589678011*0.5;
		ps[4] = vector(0.445948490915965, 0.108103018168070, 0.0);  ws[4] = 0.223381589678011*0.5;
		ps[5] = vector(0.445948490915965, 0.445948490915965, 0.0);  ws[5] = 0.223381589678011*0.5;
		standardTriGaussPoints = ps;
		standardTriGaussWeights = ws;
	}
	else
	{
		FatalErrorIn("Foam::d2GaussQuadrature::collectTriGaussQuadrature()")
			<< "No stored information for cOrder = " << cOrder << nl
			<< exit(FatalError);
	}
}


void Foam::d2GaussQuadrature::collectBarGaussQuadrature(
	const label fOrder,
	List<scalar>& standardBarGaussPoints,
	List<scalar>& standardBarGaussWeights
)
{
	if (fOrder == 1)  // accuracy grade = 1
	{
		standardBarGaussPoints.setSize(1);
		standardBarGaussWeights.setSize(1);
		List<scalar> ps(1);
		List<scalar> ws(1);
		ps[0] = 0.0;  ws[0] = 2.0;
		standardBarGaussPoints = ps;
		standardBarGaussWeights = ws;
	}
	else if (fOrder > 1 && fOrder <= 3)  // accuracy grade = 3
	{
		standardBarGaussPoints.setSize(2);
		standardBarGaussPoints.setSize(2);
		List<scalar> ps(2);
		List<scalar> ws(2);
		ps[0] = -1/Foam::sqrt(3.0);  ws[0] = 1.0;
		ps[1] = 1/Foam::sqrt(3.0);   ws[1] = 1.0;
		standardBarGaussPoints = ps;
		standardBarGaussWeights = ws;
	}
	else if (fOrder > 3 && fOrder <= 5)  // accuracy grade = 5
	{
		standardBarGaussPoints.setSize(3);
		standardBarGaussWeights.setSize(3);
		List<scalar> ps(3);
		List<scalar> ws(3);
		ps[0] = -Foam::sqrt(0.6);  ws[0] = 5.0/9.0;
		ps[1] = 0.0;               ws[1] = 8.0/9.0;
		ps[2] = Foam::sqrt(0.6);   ws[2] = 5.0/9.0;
		standardBarGaussPoints = ps;
		standardBarGaussWeights = ws;
	}
	else if (fOrder > 5 && fOrder <= 7)  // accuracy grade = 7
	{
		standardBarGaussPoints.setSize(4);
		standardBarGaussWeights.setSize(4);
		List<scalar> ps(4);
		List<scalar> ws(4);
		ps[0] = 0.861136311594053;   ws[0] = 0.347854845137454;
		ps[1] = 0.339981043584856;   ws[1] = 0.652145154862546;
		ps[2] = -0.339981043584856;  ws[2] = 0.652145154862546;
		ps[3] = -0.861136311594053;  ws[3] = 0.347854845137454;
		standardBarGaussPoints = ps;
		standardBarGaussWeights = ws;
	}
	else
	{
		FatalErrorIn("Foam::d2GaussQuadrature::collectBarGaussQuadrature()")
			<< "No stored information for fOrder = " << fOrder << nl
			<< exit(FatalError);
	}
}